import abc
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Dict, Any, List, Optional

from src.core.constants import DEFAULT_MEMORY_DB_PATH

logger = logging.getLogger(__name__)


class BaseShortTermMemory(abc.ABC):
    """Abstract base class for short-term memory storage.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    @abc.abstractmethod
    def store_sample(self, sample: Dict[str, Any]) -> int:
        """Store a sample and return row id.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 插入的行 ID
        :rtype: int
        """
        pass

    @abc.abstractmethod
    def load_samples(self, limit: int = 100, order_by: str = "timestamp DESC") -> List[Dict[str, Any]]:
        """Load samples from storage.

        :param limit: 加载样本数量限制
        :type limit: int
        :param order_by: 排序方式
        :type order_by: str
        :return: 样本列表
        :rtype: List[Dict[str, Any]]
        """
        pass


class ShortTermMemory(BaseShortTermMemory):
    """Short-term memory storage using SQLite.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    def __init__(self, db_path: str = DEFAULT_MEMORY_DB_PATH, max_samples: int = 10000, batch_size: int = 50, flush_interval: float = 5.0) -> None:
        """Initialize short-term memory.

        :param db_path: 数据库文件路径
        :type db_path: str
        :param max_samples: 最大样本数量
        :type max_samples: int
        :param batch_size: 批量写入阈值
        :type batch_size: int
        :param flush_interval: 自动刷盘间隔（秒）
        :type flush_interval: float
        :return: 无
        :rtype: None
        """
        self.db_path = db_path
        self.max_samples = self._sanitize_max_samples(max_samples)
        self.batch_size = self._sanitize_batch_size(batch_size)
        self.flush_interval = flush_interval if isinstance(flush_interval, (int, float)) and flush_interval >= 0 else 5.0
        self._lock = threading.RLock()
        self._sample_buffer: List[Dict[str, Any]] = []
        self._task_buffer: List[Dict[str, Any]] = []
        self._snapshot_buffer: List[tuple] = []
        self._last_flush_time = time.time()
        self._connect_db()

    @staticmethod
    def _sanitize_max_samples(value: Any) -> int:
        """校正 max_samples 参数，确保为正整数。

        :param value: 原始值。
        :type value: Any
        :return: 校正后的正整数。
        :rtype: int
        """
        try:
            v = int(value)
        except (TypeError, ValueError):
            return 10000
        return v if v > 0 else 10000

    @staticmethod
    def _sanitize_batch_size(value: Any) -> int:
        """校正 batch_size 参数，确保为正整数。

        :param value: 原始值。
        :type value: Any
        :return: 校正后的正整数。
        :rtype: int
        """
        try:
            v = int(value)
        except (TypeError, ValueError):
            return 50
        return v if v > 0 else 50

    def _connect_db(self) -> None:
        """Connect to the SQLite database.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_tables()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            self._handle_connect_error(e)

    def _handle_connect_error(self, e: Exception) -> None:
        """Handle database connection error.

        :param e: 异常对象
        :type e: Exception
        :return: 无
        :rtype: None
        """
        if self._is_corruption_error(e):
            logger.warning("ShortTermMemory.__init__: corrupted DB detected, recovering: %s", self.db_path)
            self._recover_corrupted_db()
        else:
            raise

    def _is_corruption_error(self, e: Exception) -> bool:
        """Check if exception indicates database corruption.

        :param e: 异常对象
        :type e: Exception
        :return: 是否为损坏错误
        :rtype: bool
        """
        msg = str(e).lower()
        return 'file is not a database' in msg or 'database disk image is malformed' in msg

    def _init_tables(self) -> None:
        """Initialize database tables.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        with self._lock:
            self._create_samples_table()
            self._create_tasks_table()
            self._create_snapshots_table()
            self._create_indexes()
            self._conn.commit()

    def _create_samples_table(self) -> None:
        """Create the samples table.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                state_json TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                actual_json TEXT NOT NULL,
                error_json TEXT NOT NULL,
                task_id TEXT,
                reward REAL,
                value_score REAL DEFAULT 0.0
            )
        """)

    def _create_tasks_table(self) -> None:
        """Create the tasks table.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_executed REAL,
                frozen INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0
            )
        """)

    def _create_snapshots_table(self) -> None:
        """Create the model_snapshots table.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                snapshot_path TEXT NOT NULL,
                mae REAL,
                coverage REAL,
                eff_score REAL,
                is_stable INTEGER DEFAULT 1
            )
        """)

    def _create_indexes(self) -> None:
        """Create database indexes.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        cursor = self._conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_timestamp ON samples(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_value ON samples(value_score DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(type)")

    def _ensure_tables(self) -> None:
        """检测并重建缺失的表（用于故障恢复）

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='samples'")
                if cursor.fetchone() is None:
                    self._init_tables()
            except sqlite3.Error:
                pass

    def _recover_corrupted_db(self) -> None:
        """数据库文件损坏时删除并重建

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        with self._lock:
            self._close_connection()
            self._remove_db_file()
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_tables()

    def _close_connection(self) -> None:
        """Close the database connection.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        try:
            self._conn.close()
        except Exception:
            pass

    def _remove_db_file(self) -> None:
        """Remove the database file.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except OSError:
            pass

    def _maybe_cleanup(self) -> None:
        """如果样本数超过上限，自动温和清理旧样本（保留80%）

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM samples")
            total = cursor.fetchone()[0]
            if total > self.max_samples:
                self.cleanup_old_samples(retention_ratio=0.8)
        except sqlite3.Error:
            pass

    def store_sample(self, sample: Dict[str, Any]) -> int:
        """Store a sample into the database.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 插入的行 ID（flush成功时返回rowid，否则返回-1）
        :rtype: int
        """
        if sample is None or not isinstance(sample, dict):
            logger.warning("ShortTermMemory.store_sample: invalid sample, skipping")
            return -1
        with self._lock:
            if not self._ensure_open():
                return -1
            # 修复Bug1a: 始终先入缓冲区，再判断是否刷盘（避免缓冲区数据卡住）
            self._sample_buffer.append(sample)
            if self._should_flush():
                # 修复Bug7: 捕获flush异常，避免崩溃agent；缓冲区数据保留待下次重试
                # 修复阶段122: 同时捕获OSError（磁盘满/IO错误），避免资源类故障崩溃agent
                try:
                    self._flush_all()
                    # flush成功后返回最后插入的rowid
                    return getattr(self, '_last_inserted_rowid', 1) or 1
                except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
                    logger.error("ShortTermMemory.store_sample: flush failed, data retained in buffer: %s", e)
            return -1

    def _should_flush(self) -> bool:
        """Check if buffers should be flushed to database.

        :param: 无
        :type: 无
        :return: 是否应该刷盘
        :rtype: bool
        """
        total = len(self._sample_buffer) + len(self._task_buffer) + len(self._snapshot_buffer)
        if total >= self.batch_size:
            return True
        if (time.time() - self._last_flush_time) >= self.flush_interval:
            return True
        return False

    def _flush_all(self) -> None:
        """Flush all buffers to database in a single transaction.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        if not self._sample_buffer and not self._task_buffer and not self._snapshot_buffer:
            return
        try:
            self._flush_samples()
            self._flush_tasks()
            self._flush_snapshots()
            self._conn.commit()
            self._last_flush_time = time.time()
            # 修复Bug1b: 只在成功时清空缓冲区（原代码在finally中无条件清空导致失败时数据丢失）
            self._sample_buffer.clear()
            self._task_buffer.clear()
            self._snapshot_buffer.clear()
            logger.debug("ShortTermMemory._flush_all: flushed samples=%s tasks=%s snapshots=%s",
                         len(self._sample_buffer), len(self._task_buffer), len(self._snapshot_buffer))
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
            self._conn.rollback()
            # 修复Bug1c: 恢复后重试flush（原代码恢复后不重试，数据在finally中被清空丢失）
            self._handle_flush_error(e)

    def _flush_samples(self) -> None:
        """Flush sample buffer to database.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        if not self._sample_buffer:
            return
        cursor = self._conn.cursor()
        for sample in self._sample_buffer:
            cursor.execute("""
                INSERT INTO samples (timestamp, state_json, prediction_json, actual_json, error_json, task_id, reward, value_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, self._sample_values(sample))
        # 跟踪最后插入的rowid，供store_sample返回
        self._last_inserted_rowid = cursor.lastrowid

    def _flush_tasks(self) -> None:
        """Flush task buffer to database.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        if not self._task_buffer:
            return
        cursor = self._conn.cursor()
        for task_data in self._task_buffer:
            cursor.execute("""
                INSERT OR REPLACE INTO tasks (id, type, name, created_at, execution_count, success_count, last_executed, frozen, avg_reward)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, self._task_values(task_data))

    def _flush_snapshots(self) -> None:
        """Flush snapshot buffer to database.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        if not self._snapshot_buffer:
            return
        cursor = self._conn.cursor()
        for snapshot_path, metrics in self._snapshot_buffer:
            cursor.execute("""
                INSERT INTO model_snapshots (timestamp, snapshot_path, mae, coverage, eff_score, is_stable)
                VALUES (?, ?, ?, ?, ?, ?)
            """, self._snapshot_values(snapshot_path, metrics))

    def _handle_flush_error(self, e: Exception) -> None:
        """Handle error during buffer flush. After recovery, retry flush.

        :param e: 异常对象
        :type e: Exception
        :return: 无
        :rtype: None
        """
        msg = str(e).lower()
        recovered = False
        if 'no such table' in msg:
            self._ensure_tables()
            recovered = True
            logger.info("ShortTermMemory._handle_flush_error: table recreated, retrying flush")
        elif self._is_corruption_msg(msg):
            logger.warning("ShortTermMemory._flush_all: corrupted DB, recovering")
            self._recover_corrupted_db()
            recovered = True
            logger.info("ShortTermMemory._handle_flush_error: DB recovered, retrying flush")
        if recovered:
            # 修复Bug1c: 恢复后重试flush，避免数据丢失
            try:
                self._flush_samples()
                self._flush_tasks()
                self._flush_snapshots()
                self._conn.commit()
                self._last_flush_time = time.time()
                self._sample_buffer.clear()
                self._task_buffer.clear()
                self._snapshot_buffer.clear()
                logger.info("ShortTermMemory._handle_flush_error: retry succeeded after recovery")
                return
            except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as retry_err:
                logger.error("ShortTermMemory._handle_flush_error: retry failed after recovery: %s", retry_err)
                raise
        raise

    def flush(self) -> None:
        """Explicitly flush all pending buffers to database.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        with self._lock:
            if not self._ensure_open():
                return
            self._flush_all()

    def _insert_sample(self, sample: Dict[str, Any]) -> int:
        """Insert a single sample into the database immediately.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 插入的行 ID
        :rtype: int
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO samples (timestamp, state_json, prediction_json, actual_json, error_json, task_id, reward, value_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, self._sample_values(sample))
        self._conn.commit()
        logger.debug("ShortTermMemory.store_sample: rowid=%s", cursor.lastrowid)
        return cursor.lastrowid

    def _sample_values(self, sample: Dict[str, Any]) -> tuple:
        """Extract values from sample for insertion.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 插入值元组
        :rtype: tuple
        """
        return (
            sample.get('timestamp', time.time()),
            *self._sample_json_values(sample),
            sample.get('task_id'),
            sample.get('reward', 0.0),
            sample.get('value_score', 0.0),
        )

    def _sample_json_values(self, sample: Dict[str, Any]) -> tuple:
        """Extract JSON values from sample.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: JSON 值元组
        :rtype: tuple
        """
        return (
            json.dumps(sample.get('state', {})),
            json.dumps(sample.get('predictions', {})),
            json.dumps(sample.get('actuals', {})),
            json.dumps(sample.get('errors', {})),
        )

    def _handle_store_sample_error(self, e: Exception, sample: Dict[str, Any]) -> int:
        """Handle error during sample storage.

        :param e: 异常对象
        :type e: Exception
        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 插入的行 ID
        :rtype: int
        """
        msg = str(e).lower()
        if 'no such table' in msg:
            return self._retry_store_after_ensure_tables(sample)
        if self._is_corruption_msg(msg):
            return self._retry_store_after_recovery(sample)
        self._conn.rollback()
        raise

    def _retry_store_after_ensure_tables(self, sample: Dict[str, Any]) -> int:
        """Retry storing sample after ensuring tables exist.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 插入的行 ID
        :rtype: int
        """
        self._ensure_tables()
        return self.store_sample(sample)

    def _retry_store_after_recovery(self, sample: Dict[str, Any]) -> int:
        """Retry storing sample after database recovery.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 插入的行 ID
        :rtype: int
        """
        logger.warning("ShortTermMemory.store_sample: corrupted DB, recovering")
        self._recover_corrupted_db()
        return self.store_sample(sample)

    def _is_corruption_msg(self, msg: str) -> bool:
        """Check if message indicates database corruption.

        :param msg: 错误消息
        :type msg: str
        :return: 是否为损坏消息
        :rtype: bool
        """
        return 'file is not a database' in msg or 'database disk image is malformed' in msg

    def load_samples(self, limit: int = 100, order_by: str = "timestamp DESC") -> List[Dict[str, Any]]:
        """Load samples from the database.

        :param limit: 加载样本数量限制
        :type limit: int
        :param order_by: 排序方式
        :type order_by: str
        :return: 样本列表
        :rtype: List[Dict[str, Any]]
        """
        with self._lock:
            if not self._ensure_open():
                return []
            try:
                return self._fetch_samples(limit, order_by)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                return self._handle_load_samples_error(e)

    def _fetch_samples(self, limit: int, order_by: str) -> List[Dict[str, Any]]:
        """Fetch samples from the database.

        :param limit: 加载样本数量限制
        :type limit: int
        :param order_by: 排序方式
        :type order_by: str
        :return: 样本列表
        :rtype: List[Dict[str, Any]]
        """
        cursor = self._conn.cursor()
        cursor.execute(f"""
            SELECT * FROM samples ORDER BY {order_by} LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [self._parse_sample_row(dict(zip(columns, row))) for row in rows]

    def _parse_sample_row(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a sample row from the database.

        :param sample: 原始样本字典
        :type sample: Dict[str, Any]
        :return: 解析后的样本字典
        :rtype: Dict[str, Any]
        """
        for key in ['state_json', 'prediction_json', 'actual_json', 'error_json']:
            sample[key] = json.loads(sample[key])
        sample['errors'] = sample.get('error_json', {})
        sample['state'] = sample.get('state_json', {})
        sample['predictions'] = sample.get('prediction_json', {})
        sample['actuals'] = sample.get('actual_json', {})
        sample['mae'] = self._compute_mae(sample.get('errors', {}))
        return sample

    def _compute_mae(self, error_dict: Dict[str, Any]) -> float:
        """Compute mean absolute error from error dictionary.

        :param error_dict: 误差字典
        :type error_dict: Dict[str, Any]
        :return: 平均绝对误差
        :rtype: float
        """
        if error_dict:
            return sum(abs(v) for v in error_dict.values()) / len(error_dict)
        return 0.0

    def _handle_load_samples_error(self, e: Exception) -> List[Dict[str, Any]]:
        """Handle error during sample loading.

        :param e: 异常对象
        :type e: Exception
        :return: 空样本列表
        :rtype: List[Dict[str, Any]]
        """
        msg = str(e).lower()
        if 'no such table' in msg:
            self._ensure_tables()
            return []
        if 'file is not a database' in msg or 'database disk image is malformed' in msg:
            self._recover_corrupted_db()
            return []
        raise

    def load_high_value_samples(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load high-value samples ordered by value_score.

        :param limit: 加载样本数量限制
        :type limit: int
        :return: 高价值样本列表
        :rtype: List[Dict[str, Any]]
        """
        return self.load_samples(limit=limit, order_by="value_score DESC")

    def update_value_score(self, sample_id: int, value_score: float) -> None:
        """Update the value score of a sample.

        :param sample_id: 样本 ID
        :type sample_id: int
        :param value_score: 价值分数
        :type value_score: float
        :return: 无
        :rtype: None
        """
        with self._lock:
            if not self._ensure_open():
                return
            try:
                cursor = self._conn.cursor()
                cursor.execute("UPDATE samples SET value_score = ? WHERE id = ?", (value_score, sample_id))
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
                raise

    def store_task(self, task_data: Dict[str, Any]) -> None:
        """Store or replace a task record.

        :param task_data: 任务数据
        :type task_data: Dict[str, Any]
        :return: 无
        :rtype: None
        """
        with self._lock:
            if not self._ensure_open():
                return
            if self._should_flush():
                self._insert_task(task_data)
                return
            self._task_buffer.append(task_data)

    def _insert_task(self, task_data: Dict[str, Any]) -> None:
        """Insert or replace a task record.

        :param task_data: 任务数据
        :type task_data: Dict[str, Any]
        :return: 无
        :rtype: None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tasks (id, type, name, created_at, execution_count, success_count, last_executed, frozen, avg_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, self._task_values(task_data))
        self._conn.commit()

    def _task_values(self, task_data: Dict[str, Any]) -> tuple:
        """Extract values from task data for insertion.

        :param task_data: 任务数据
        :type task_data: Dict[str, Any]
        :return: 插入值元组
        :rtype: tuple
        """
        return (
            task_data['id'],
            task_data['type'],
            task_data['name'],
            task_data['created_at'],
            *self._task_optional_values(task_data),
        )

    def _task_optional_values(self, task_data: Dict[str, Any]) -> tuple:
        """Extract optional values from task data.

        :param task_data: 任务数据
        :type task_data: Dict[str, Any]
        :return: 可选值元组
        :rtype: tuple
        """
        return (
            task_data.get('execution_count', 0),
            task_data.get('success_count', 0),
            task_data.get('last_executed'),
            1 if task_data.get('frozen', False) else 0,
            task_data.get('avg_reward', 0.0),
        )

    def store_snapshot(self, snapshot_path: str, metrics: Dict[str, float]) -> int:
        """Store a model snapshot record.

        :param snapshot_path: 快照路径
        :type snapshot_path: str
        :param metrics: 指标数据
        :type metrics: Dict[str, float]
        :return: 插入的行 ID
        :rtype: int
        """
        with self._lock:
            if not self._ensure_open():
                return -1
            if self._should_flush():
                return self._insert_snapshot(snapshot_path, metrics)
            self._snapshot_buffer.append((snapshot_path, metrics))
            return -1

    def _insert_snapshot(self, snapshot_path: str, metrics: Dict[str, float]) -> int:
        """Insert a model snapshot record.

        :param snapshot_path: 快照路径
        :type snapshot_path: str
        :param metrics: 指标数据
        :type metrics: Dict[str, float]
        :return: 插入的行 ID
        :rtype: int
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO model_snapshots (timestamp, snapshot_path, mae, coverage, eff_score, is_stable)
            VALUES (?, ?, ?, ?, ?, ?)
        """, self._snapshot_values(snapshot_path, metrics))
        self._conn.commit()
        return cursor.lastrowid

    def _snapshot_values(self, snapshot_path: str, metrics: Dict[str, float]) -> tuple:
        """Extract values for snapshot insertion.

        :param snapshot_path: 快照路径
        :type snapshot_path: str
        :param metrics: 指标数据
        :type metrics: Dict[str, float]
        :return: 插入值元组
        :rtype: tuple
        """
        return (
            time.time(),
            snapshot_path,
            metrics.get('mae'),
            metrics.get('coverage'),
            metrics.get('eff_score'),
            1 if metrics.get('is_stable', True) else 0,
        )

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the latest model snapshot record.

        :param: 无
        :type: 无
        :return: 最新快照记录或 None
        :rtype: Optional[Dict[str, Any]]
        """
        with self._lock:
            if not self._ensure_open():
                return None
            return self._fetch_latest_snapshot()

    def _fetch_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Fetch the latest model snapshot record.

        :param: 无
        :type: 无
        :return: 最新快照记录或 None
        :rtype: Optional[Dict[str, Any]]
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM model_snapshots ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None

    def cleanup_old_samples(self, retention_ratio: float = 0.3) -> int:
        """Cleanup old samples retaining top value scores.

        :param retention_ratio: 保留比例
        :type retention_ratio: float
        :return: 删除的样本数量
        :rtype: int
        """
        with self._lock:
            if not self._ensure_open():
                return 0
            try:
                return self._delete_old_samples(retention_ratio)
            except sqlite3.Error:
                self._conn.rollback()
                raise

    def _delete_old_samples(self, retention_ratio: float) -> int:
        """Delete old samples based on retention ratio.

        :param retention_ratio: 保留比例
        :type retention_ratio: float
        :return: 删除的样本数量
        :rtype: int
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM samples")
        total = cursor.fetchone()[0]
        if total <= self.max_samples:
            return 0
        return self._execute_sample_cleanup(cursor, total, retention_ratio)

    def _execute_sample_cleanup(self, cursor, total: int, retention_ratio: float) -> int:
        """Execute sample cleanup.

        :param cursor: 数据库游标
        :type cursor: sqlite3.Cursor
        :param total: 样本总数
        :type total: int
        :param retention_ratio: 保留比例
        :type retention_ratio: float
        :return: 删除的样本数量
        :rtype: int
        """
        keep_count = int(total * retention_ratio)
        cursor.execute(self._cleanup_sql(), (keep_count,))
        return self._finalize_cleanup(cursor)

    def _cleanup_sql(self) -> str:
        """Return the cleanup SQL statement.

        :param: 无
        :type: 无
        :return: 清理 SQL 语句
        :rtype: str
        """
        return """
            DELETE FROM samples WHERE id NOT IN (
                SELECT id FROM samples ORDER BY value_score DESC LIMIT ?
            )
        """

    def _finalize_cleanup(self, cursor) -> int:
        """Finalize cleanup and return deleted count.

        :param cursor: 数据库游标
        :type cursor: sqlite3.Cursor
        :return: 删除的样本数量
        :rtype: int
        """
        deleted = cursor.rowcount
        self._conn.commit()
        self._conn.execute("VACUUM")
        return deleted

    def get_sample_count(self) -> int:
        """Return the total number of samples.

        :param: 无
        :type: 无
        :return: 样本总数
        :rtype: int
        """
        with self._lock:
            if not self._ensure_open():
                return 0
            try:
                return self._execute_count_samples()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                return self._handle_count_samples_error(e)

    def _execute_count_samples(self) -> int:
        """Execute count samples query.

        :param: 无
        :type: 无
        :return: 样本总数
        :rtype: int
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM samples")
        return cursor.fetchone()[0]

    def _handle_count_samples_error(self, e: Exception) -> int:
        """Handle error during count samples.

        :param e: 异常对象
        :type e: Exception
        :return: 样本总数
        :rtype: int
        """
        msg = str(e).lower()
        if 'no such table' in msg:
            self._ensure_tables()
            return 0
        if 'file is not a database' in msg or 'database disk image is malformed' in msg:
            self._recover_corrupted_db()
            return 0
        raise

    def close(self) -> None:
        """Close the database connection.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        with self._lock:
            if not getattr(self, '_closed', False):
                # 修复阶段122: close时flush失败不应阻止连接关闭
                try:
                    self._flush_all()
                except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
                    logger.error("ShortTermMemory.close: flush failed during close: %s", e)
                try:
                    self._conn.close()
                except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as e:
                    logger.error("ShortTermMemory.close: conn.close failed: %s", e)
                self._closed = True

    def _ensure_open(self) -> bool:
        """Check if the database connection is open.

        :param: 无
        :type: 无
        :return: 连接是否打开
        :rtype: bool
        """
        return not getattr(self, '_closed', False)
