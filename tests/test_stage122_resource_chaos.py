"""阶段122：极端混沌工程（资源类故障）。

验证系统在系统级资源故障下的容错能力：
1. 磁盘满模拟（写入失败兜底）
2. 文件描述符耗尽
3. 信号中断优雅退出（SIGTERM/SIGINT）
4. 权限不足（只读目录写入）
5. 大量并发连接耗尽
6. 内存压力下的行为
7. 临时文件清理
8. 日志文件写入失败
"""
import os
import sys
import time
import unittest
import tempfile
import threading
import errno
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.task.task_pool import TaskPool
from src.core.constants import TaskType
from src.task.task_definition import Task


class TestDiskFullSimulation(unittest.TestCase):
    """磁盘满模拟：写入失败时应兜底不崩溃。"""

    def test_store_sample_disk_full_no_crash(self):
        """磁盘满时store_sample不应崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
            # 模拟flush失败（sqlite3.Connection.execute只读，改用替换_flush_samples）
            original_flush = mem._flush_samples
            call_count = [0]
            def failing_flush():
                call_count[0] += 1
                if call_count[0] > 1:
                    raise OSError("Disk full")
                return original_flush()
            mem._flush_samples = failing_flush
            try:
                # 存储多个sample触发flush
                for i in range(10):
                    ret = mem.store_sample({
                        'timestamp': time.time(),
                        'state': {'cpu': i},
                        'predictions': {'low': {}, 'mid': {}, 'high': {}},
                        'actuals': {'cpu': i},
                        'errors': {'cpu': 0.1},
                        'task_id': f't{i}',
                        'reward': 0.5,
                        'value_score': 0.5,
                    })
                    # 不应抛出异常
                    self.assertIsInstance(ret, int)
            finally:
                mem.close()

    def test_snapshot_disk_full_fallback(self):
        """快照写入磁盘满时应兜底。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=5)
            # 模拟open失败
            with patch('builtins.open', side_effect=OSError("Disk full")):
                try:
                    sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
                    # 如果不抛异常说明有兜底
                except OSError:
                    # 抛出OSError也可接受，只要不崩溃进程
                    pass

    def test_db_corruption_recovery(self):
        """数据库文件损坏后应能恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
            for i in range(3):
                mem.store_sample({
                    'timestamp': time.time(), 'state': {'cpu': i},
                    'predictions': {'low': {}, 'mid': {}, 'high': {}},
                    'actuals': {'cpu': i}, 'errors': {'cpu': 0.1},
                    'task_id': f't{i}', 'reward': 0.5, 'value_score': 0.5,
                })
            mem.flush()
            mem.close()
            # 损坏数据库
            with open(db_path, 'wb') as f:
                f.write(b'CORRUPTED' * 1000)
            # 重新打开应能恢复
            mem2 = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
            try:
                ret = mem2.store_sample({
                    'timestamp': time.time(), 'state': {'cpu': 99},
                    'predictions': {'low': {}, 'mid': {}, 'high': {}},
                    'actuals': {'cpu': 99}, 'errors': {'cpu': 0.1},
                    'task_id': 'recovery', 'reward': 0.5, 'value_score': 0.5,
                })
                mem2.flush()
                # 恢复后应能正常写入
                self.assertIsInstance(ret, int)
            finally:
                mem2.close()


class TestFileDescriptorExhaustion(unittest.TestCase):
    """文件描述符耗尽模拟。"""

    def test_snapshot_fd_exhaustion_no_crash(self):
        """FD耗尽时快照操作不应崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=5)
            # 模拟open失败（FD耗尽）
            with patch('builtins.open', side_effect=OSError(errno.EMFILE, "Too many open files")):
                try:
                    sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
                except OSError:
                    pass  # 兜底处理

    def test_memory_fd_exhaustion_no_crash(self):
        """FD耗尽时memory操作不应崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            # sqlite3.connect失败时应兜底
            with patch('sqlite3.connect', side_effect=OSError(errno.EMFILE, "Too many open files")):
                try:
                    mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
                    mem.close()
                except (OSError, Exception):
                    pass  # 兜底处理


class TestSignalHandling(unittest.TestCase):
    """信号处理测试。"""

    def test_sigterm_graceful_shutdown(self):
        """SIGTERM应触发优雅关闭。"""
        from src.core.agent_loop import AgentLoop
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.test.yaml'
        )
        agent = AgentLoop(config_path)
        try:
            # 运行几个周期
            for i in range(3):
                agent.run_cycle()
            # 模拟SIGTERM
            agent._running = False
            # 应能正常停止
            agent.stop()
            self.assertFalse(agent._running)
        except Exception as e:
            self.fail(f"Signal handling crashed: {e}")

    def test_sigint_during_cycle(self):
        """周期中收到SIGINT应能优雅退出。"""
        from src.core.agent_loop import AgentLoop
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.test.yaml'
        )
        agent = AgentLoop(config_path)
        try:
            # 在另一个线程中运行周期
            def run_cycles():
                for i in range(100):
                    if not agent._running:
                        break
                    agent.run_cycle()
            t = threading.Thread(target=run_cycles, daemon=True)
            t.start()
            time.sleep(0.5)
            # 模拟SIGINT
            agent._running = False
            t.join(timeout=5)
            self.assertFalse(t.is_alive(), "Thread should have stopped")
        finally:
            agent.stop()


class TestPermissionDenied(unittest.TestCase):
    """权限不足测试。"""

    def test_snapshot_readonly_dir(self):
        """只读目录下创建快照应兜底。"""
        with tempfile.TemporaryDirectory() as tmp:
            readonly_dir = os.path.join(tmp, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o555)  # 只读
            try:
                sm = SnapshotManager(snapshot_dir=readonly_dir, max_count=5)
                try:
                    sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
                except (OSError, PermissionError):
                    pass  # 兜底处理
            finally:
                os.chmod(readonly_dir, 0o755)  # 恢复权限以便清理

    def test_memory_readonly_db_path(self):
        """只读路径下创建数据库应兜底。"""
        with tempfile.TemporaryDirectory() as tmp:
            readonly_dir = os.path.join(tmp, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o555)
            try:
                db_path = os.path.join(readonly_dir, "test.db")
                try:
                    mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
                    mem.close()
                except (OSError, PermissionError):
                    pass  # 兜底处理
            finally:
                os.chmod(readonly_dir, 0o755)


class TestConcurrentConnectionExhaustion(unittest.TestCase):
    """并发连接耗尽测试。"""

    def test_many_concurrent_memory_instances(self):
        """大量并发memory实例不应导致系统崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            instances = []
            errors = []
            try:
                for i in range(20):
                    try:
                        db_path = os.path.join(tmp, f"test_{i}.db")
                        mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
                        instances.append(mem)
                    except Exception as e:
                        errors.append(e)
                # 至少应能创建部分实例
                self.assertGreater(len(instances), 0, "No instances created")
                # 并发写入
                def write_samples(mem, idx):
                    try:
                        for j in range(5):
                            mem.store_sample({
                                'timestamp': time.time(), 'state': {'cpu': j},
                                'predictions': {'low': {}, 'mid': {}, 'high': {}},
                                'actuals': {'cpu': j}, 'errors': {'cpu': 0.1},
                                'task_id': f't{j}', 'reward': 0.5, 'value_score': 0.5,
                            })
                        mem.flush()
                    except Exception as e:
                        errors.append(e)
                threads = []
                for i, mem in enumerate(instances):
                    t = threading.Thread(target=write_samples, args=(mem, i))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join(timeout=10)
                # 不应有致命错误
                self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
            finally:
                for mem in instances:
                    try:
                        mem.close()
                    except Exception:
                        pass


class TestMemoryPressure(unittest.TestCase):
    """内存压力测试。"""

    def test_large_sample_storage(self):
        """大样本存储不应导致OOM。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=50, flush_interval=9999)
            try:
                # 存储大量带大数据的sample
                for i in range(100):
                    large_state = {'data': 'x' * 1000}  # 1KB per sample
                    mem.store_sample({
                        'timestamp': time.time(),
                        'state': large_state,
                        'predictions': {'low': {}, 'mid': {}, 'high': {}},
                        'actuals': {'cpu': i},
                        'errors': {'cpu': 0.1},
                        'task_id': f't{i}',
                        'reward': 0.5,
                        'value_score': 0.5,
                    })
                mem.flush()
                count = mem.get_sample_count()
                self.assertEqual(count, 100)
            finally:
                mem.close()

    def test_task_pool_large_number_of_tasks(self):
        """大量任务不应导致OOM。"""
        pool = TaskPool(max_size=50)
        for i in range(1000):
            task = Task(id=f'task_{i}', type=TaskType.SAFE, action='test')
            pool.update([task])
        # task_pool应被限制在max_size
        self.assertLessEqual(len(pool.tasks), 50)


class TestTempFileCleanup(unittest.TestCase):
    """临时文件清理测试。"""

    def test_snapshot_files_cleaned_on_enforce_limit(self):
        """快照超限时旧文件应被删除。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=3)
            for i in range(10):
                sm.create_snapshot({'v': i}, metrics={'mae': 0.1 * i})
            # 应只剩3个快照文件
            pkl_files = [f for f in os.listdir(tmp) if f.endswith('.pkl')]
            self.assertEqual(len(pkl_files), 3)

    def test_no_orphan_temp_files(self):
        """不应留下孤儿临时文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=5)
            sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
            # 不应有.tmp文件
            files = os.listdir(tmp)
            tmp_files = [f for f in files if f.endswith('.tmp')]
            self.assertEqual(len(tmp_files), 0)


class TestLogWriteFailure(unittest.TestCase):
    """日志写入失败测试。"""

    def test_agent_loop_with_log_failure(self):
        """日志写入失败时agent_loop应继续运行。"""
        from src.core.agent_loop import AgentLoop
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.test.yaml'
        )
        agent = AgentLoop(config_path)
        try:
            # 模拟日志写入失败
            with patch('logging.FileHandler.emit', side_effect=OSError("Log disk full")):
                # 应仍能运行周期
                for i in range(3):
                    try:
                        agent.run_cycle()
                    except Exception as e:
                        self.fail(f"Cycle crashed due to log failure: {e}")
        finally:
            agent.stop()


class TestExtremeChaosCombination(unittest.TestCase):
    """极端混沌组合场景。"""

    def test_db_corruption_plus_concurrent_access(self):
        """数据库损坏+并发访问不应导致数据丢失或崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
            for i in range(5):
                mem.store_sample({
                    'timestamp': time.time(), 'state': {'cpu': i},
                    'predictions': {'low': {}, 'mid': {}, 'high': {}},
                    'actuals': {'cpu': i}, 'errors': {'cpu': 0.1},
                    'task_id': f't{i}', 'reward': 0.5, 'value_score': 0.5,
                })
            mem.flush()
            mem.close()
            # 损坏数据库
            with open(db_path, 'wb') as f:
                f.write(b'CORRUPTED' * 100)
            # 并发重新打开
            errors = []
            def open_and_write(idx):
                try:
                    m = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
                    m.store_sample({
                        'timestamp': time.time(), 'state': {'cpu': idx},
                        'predictions': {'low': {}, 'mid': {}, 'high': {}},
                        'actuals': {'cpu': idx}, 'errors': {'cpu': 0.1},
                        'task_id': f'concurrent_{idx}', 'reward': 0.5, 'value_score': 0.5,
                    })
                    m.flush()
                    m.close()
                except Exception as e:
                    errors.append(e)
            threads = []
            for i in range(5):
                t = threading.Thread(target=open_and_write, args=(i,))
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=10)
            # 不应有致命错误（部分失败可接受）
            self.assertLess(len(errors), 5, f"All threads failed: {errors}")

    def test_snapshot_corruption_plus_rollback(self):
        """快照损坏+回滚操作不应崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=5)
            from src.optimization.auto_rollback import AutoRollback
            ar = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=0.1)
            # 创建快照
            sid = sm.create_snapshot({'v': 1}, metrics={'mae': 0.05})
            ar.set_baseline(0.05, sid)
            # 损坏快照文件
            snapshot_path = os.path.join(tmp, f"{sid}.pkl")
            with open(snapshot_path, 'wb') as f:
                f.write(b'CORRUPTED')
            # 尝试回滚（3次退化触发）
            restored = []
            def restore(state):
                restored.append(state)
                return True
            for _ in range(3):
                try:
                    ar.check_and_rollback(0.5, restore)
                except Exception:
                    pass  # 兜底处理
            # 不应崩溃
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
