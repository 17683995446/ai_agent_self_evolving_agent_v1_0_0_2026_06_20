import abc
import logging
import os
import pickle
import threading
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseSnapshotManager(abc.ABC):
    def __init__(self, snapshot_dir: str = "snapshots", max_count: int = 10) -> None:
        """初始化快照管理器。

        :param snapshot_dir: 快照存储目录路径
        :type snapshot_dir: str
        :param max_count: 最大保留快照数量
        :type max_count: int
        :return: 无
        :rtype: None
        """
        self.snapshot_dir = snapshot_dir
        # 修复Bug4: 规范化负max_count为最小值1，避免min([])崩溃
        self.max_count = max(1, max_count) if max_count > 0 else 1
        self._ensure_snapshot_dir()
        self._snapshots: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._load_index()

    def _ensure_snapshot_dir(self) -> None:
        """确保快照目录存在，处理目录被替换为文件的情况。"""
        if os.path.isfile(self.snapshot_dir):
            os.remove(self.snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)

    @abc.abstractmethod
    def create_snapshot(self, model_state: Any, tag: str = "auto",
                        metrics: Optional[Dict[str, float]] = None) -> str:
        """创建快照。

        :param model_state: 模型状态对象
        :type model_state: Any
        :param tag: 快照标签
        :type tag: str
        :param metrics: 可选的指标字典
        :type metrics: Optional[Dict[str, float]]
        :return: 快照唯一标识符
        :rtype: str
        """

    @abc.abstractmethod
    def load_snapshot(self, snapshot_id: str) -> Optional[Any]:
        """加载快照。

        :param snapshot_id: 快照唯一标识符
        :type snapshot_id: str
        :return: 模型状态对象，未找到时返回None
        :rtype: Optional[Any]
        """

    @abc.abstractmethod
    def get_latest_snapshot(self, tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取最新快照。

        :param tag: 可选的标签过滤
        :type tag: Optional[str]
        :return: 最新快照的元数据字典，无快照时返回None
        :rtype: Optional[Dict[str, Any]]
        """

    @abc.abstractmethod
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有快照。

        :param: 无
        :type: 无
        :return: 快照元数据列表
        :rtype: List[Dict[str, Any]]
        """

    @abc.abstractmethod
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除快照。

        :param snapshot_id: 快照唯一标识符
        :type snapshot_id: str
        :return: 删除是否成功
        :rtype: bool
        """

    @abc.abstractmethod
    def verify_and_repair(self) -> Dict[str, Any]:
        """校验并修复index与磁盘文件的一致性。

        :param: 无
        :type: 无
        :return: 修复结果字典
        :rtype: Dict[str, Any]
        """

    @abc.abstractmethod
    def compare_snapshots(self, id1: str, id2: str) -> Dict[str, Any]:
        """比较两个快照。

        :param id1: 第一个快照标识符
        :type id1: str
        :param id2: 第二个快照标识符
        :type id2: str
        :return: 比较结果字典
        :rtype: Dict[str, Any]
        """

    def _load_index(self) -> None:
        index_path = os.path.join(self.snapshot_dir, "index.json")
        if not os.path.exists(index_path):
            return
        import json
        try:
            with open(index_path, 'r') as f:
                self._snapshots = json.load(f)
            self._verify_index()
        except (json.JSONDecodeError, IOError):
            self._snapshots = []

    def _save_index(self) -> None:
        import json
        index_path = os.path.join(self.snapshot_dir, "index.json")
        temp_path = index_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(self._snapshots, f, indent=2)
        os.replace(temp_path, index_path)

    def _verify_index(self) -> int:
        """校验index与磁盘文件一致性，移除无效记录，返回移除数量"""
        valid = []
        removed = 0
        for entry in self._snapshots:
            if self._is_entry_valid(entry):
                valid.append(entry)
            else:
                removed += 1
        self._snapshots = valid
        if removed > 0:
            self._save_index()
        return removed

    def _is_entry_valid(self, entry: Dict[str, Any]) -> bool:
        filepath = entry.get('filepath', '')
        return bool(filepath and os.path.exists(filepath))


class SnapshotManager(BaseSnapshotManager):
    def create_snapshot(self, model_state: Any, tag: str = "auto",
                        metrics: Optional[Dict[str, float]] = None) -> str:
        """创建快照。

        :param model_state: 模型状态对象
        :type model_state: Any
        :param tag: 快照标签
        :type tag: str
        :param metrics: 可选的指标字典
        :type metrics: Optional[Dict[str, float]]
        :return: 快照唯一标识符
        :rtype: str
        """
        with self._lock:
            snapshot_id = self._write_snapshot_file(model_state, tag)
            entry = self._build_snapshot_entry(snapshot_id, tag, metrics)
            self._snapshots.append(entry)
            self._enforce_limit()
            self._save_index()
            return snapshot_id

    def _write_snapshot_file(self, model_state: Any, tag: str) -> str:
        # 修复阶段116发现的bug：目录可能被删除，写入前确保存在
        self._ensure_snapshot_dir()
        timestamp = time.time()
        snapshot_id = f"{tag}_{timestamp:.6f}"
        filepath = os.path.join(self.snapshot_dir, f"{snapshot_id}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        logger.info("SnapshotManager.create_snapshot: id=%s path=%s", snapshot_id, filepath)
        return snapshot_id

    def _build_snapshot_entry(self, snapshot_id: str, tag: str,
                              metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
        timestamp = float(snapshot_id.rsplit('_', 1)[1])
        filepath = os.path.join(self.snapshot_dir, f"{snapshot_id}.pkl")
        return {
            'id': snapshot_id,
            'tag': tag,
            'timestamp': timestamp,
            'filepath': filepath,
            'metrics': metrics or {},
        }

    def load_snapshot(self, snapshot_id: str) -> Optional[Any]:
        """加载快照。

        :param snapshot_id: 快照唯一标识符
        :type snapshot_id: str
        :return: 模型状态对象，未找到时返回None
        :rtype: Optional[Any]
        """
        with self._lock:
            entry = self._find_snapshot_entry(snapshot_id)
            if entry is None:
                logger.warning("SnapshotManager.load_snapshot: snapshot not found id=%s", snapshot_id)
                return None
            filepath = entry['filepath']
            if not os.path.exists(filepath):
                logger.warning("SnapshotManager.load_snapshot: snapshot file missing id=%s path=%s",
                               snapshot_id, filepath)
                return None
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            logger.info("SnapshotManager.load_snapshot: loaded id=%s path=%s", snapshot_id, filepath)
            return state

    def _find_snapshot_entry(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._snapshots:
            if entry['id'] == snapshot_id:
                return entry
        return None

    def get_latest_snapshot(self, tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取最新快照。

        :param tag: 可选的标签过滤
        :type tag: Optional[str]
        :return: 最新快照的元数据字典，无快照时返回None
        :rtype: Optional[Dict[str, Any]]
        """
        with self._lock:
            candidates = self._snapshots
            if tag:
                candidates = [s for s in candidates if s['tag'] == tag]
            if not candidates:
                return None
            return max(candidates, key=lambda s: s['timestamp'])

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有快照。

        :param: 无
        :type: 无
        :return: 快照元数据列表
        :rtype: List[Dict[str, Any]]
        """
        with self._lock:
            return list(self._snapshots)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除快照。

        :param snapshot_id: 快照唯一标识符
        :type snapshot_id: str
        :return: 删除是否成功
        :rtype: bool
        """
        with self._lock:
            for i, entry in enumerate(self._snapshots):
                if entry['id'] == snapshot_id:
                    self._delete_snapshot_file(entry)
                    self._snapshots.pop(i)
                    self._save_index()
                    logger.info("SnapshotManager.delete_snapshot: deleted id=%s", snapshot_id)
                    return True
            logger.warning("SnapshotManager.delete_snapshot: snapshot not found id=%s", snapshot_id)
            return False

    def _delete_snapshot_file(self, entry: Dict[str, Any]) -> None:
        if os.path.exists(entry['filepath']):
            try:
                os.remove(entry['filepath'])
            except OSError:
                pass

    def _enforce_limit(self) -> None:
        # 修复Bug4: 增加空列表保护，避免min([])崩溃
        while self._snapshots and len(self._snapshots) > self.max_count:
            oldest = min(self._snapshots, key=lambda s: s['timestamp'])
            self._remove_snapshot_entry(oldest)
            self._save_index()

    def _remove_snapshot_entry(self, entry: Dict[str, Any]) -> None:
        if os.path.exists(entry['filepath']):
            try:
                os.remove(entry['filepath'])
            except OSError:
                pass
        self._snapshots.remove(entry)

    def verify_and_repair(self) -> Dict[str, Any]:
        """校验并修复index与磁盘文件的一致性。

        :param: 无
        :type: 无
        :return: 修复结果字典
        :rtype: Dict[str, Any]
        """
        with self._lock:
            removed = self._verify_index()
            orphaned = self._remove_orphaned_files()
            return self._build_repair_result(removed, orphaned)

    def _remove_orphaned_files(self) -> int:
        orphaned = 0
        for fname in os.listdir(self.snapshot_dir):
            if not fname.endswith('.pkl'):
                continue
            if self._remove_if_orphaned(fname):
                orphaned += 1
        return orphaned

    def _remove_if_orphaned(self, fname: str) -> bool:
        fpath = os.path.join(self.snapshot_dir, fname)
        if any(s.get('filepath') == fpath for s in self._snapshots):
            return False
        try:
            os.remove(fpath)
            return True
        except OSError:
            return False

    def _build_repair_result(self, removed: int, orphaned: int) -> Dict[str, Any]:
        return {
            'invalid_index_entries_removed': removed,
            'orphaned_files_removed': orphaned,
            'valid_snapshots': len(self._snapshots),
        }

    def compare_snapshots(self, id1: str, id2: str) -> Dict[str, Any]:
        """比较两个快照。

        :param id1: 第一个快照标识符
        :type id1: str
        :param id2: 第二个快照标识符
        :type id2: str
        :return: 比较结果字典
        :rtype: Dict[str, Any]
        """
        with self._lock:
            s1 = self._find_snapshot_by_id(id1)
            s2 = self._find_snapshot_by_id(id2)
            if not s1 or not s2:
                return {'error': 'snapshot not found'}
            diffs = self._compute_metric_diffs(s1, s2)
            return self._build_compare_result(id1, id2, s1, s2, diffs)

    def _find_snapshot_by_id(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        return next((s for s in self._snapshots if s['id'] == snapshot_id), None)

    def _compute_metric_diffs(self, s1: Dict[str, Any], s2: Dict[str, Any]) -> Dict[str, float]:
        m1 = s1.get('metrics', {})
        m2 = s2.get('metrics', {})
        diffs: Dict[str, float] = {}
        for key in set(m1.keys()) | set(m2.keys()):
            diffs[key] = self._metric_diff(m1, m2, key)
        return diffs

    def _metric_diff(self, m1: Dict[str, float], m2: Dict[str, float], key: str) -> float:
        v1 = m1.get(key, 0.0)
        v2 = m2.get(key, 0.0)
        return round(v2 - v1, 6)

    def _build_compare_result(self, id1: str, id2: str,
                              s1: Dict[str, Any], s2: Dict[str, Any],
                              diffs: Dict[str, float]) -> Dict[str, Any]:
        return {
            'snapshot1': id1,
            'snapshot2': id2,
            'diffs': diffs,
            'time_delta': round(s2['timestamp'] - s1['timestamp'], 2),
        }
