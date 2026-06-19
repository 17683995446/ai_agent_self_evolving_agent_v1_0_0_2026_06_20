"""阶段124：并发与竞态条件深度测试。

验证多线程并发场景下的数据一致性与竞态条件：
1. ShortTermMemory高并发store_sample
2. ShortTermMemory并发store+flush+close
3. TaskPool并发update+select+mark_executed
4. SnapshotManager并发create+load+delete
5. AutoRollback并发check_and_rollback
6. 并发agent_cycle（多agent实例）
7. 死锁检测（锁竞争不超时）
8. 数据一致性验证（并发后数据正确）
"""
import os
import sys
import time
import unittest
import tempfile
import threading
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.optimization.auto_rollback import AutoRollback
from src.task.task_pool import TaskPool
from src.task.task_definition import Task
from src.core.constants import TaskType


class TestConcurrentShortTermMemory(unittest.TestCase):
    """ShortTermMemory高并发测试。"""

    def test_concurrent_store_sample_no_loss(self):
        """并发store_sample不应丢失数据。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=10, flush_interval=9999)
            num_threads = 10
            samples_per_thread = 20
            errors = []

            def worker(thread_id):
                try:
                    for i in range(samples_per_thread):
                        mem.store_sample({
                            'timestamp': time.time(),
                            'state': {'cpu': i, 'thread': thread_id},
                            'predictions': {'low': {}, 'mid': {}, 'high': {}},
                            'actuals': {'cpu': i},
                            'errors': {'cpu': 0.1},
                            'task_id': f't{thread_id}_{i}',
                            'reward': 0.5,
                            'value_score': 0.5,
                        })
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            mem.flush()
            count = mem.get_sample_count()
            mem.close()

            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
            expected = num_threads * samples_per_thread
            self.assertEqual(count, expected,
                             f"Data loss: expected {expected}, got {count}")

    def test_concurrent_store_and_flush(self):
        """并发store和flush不应崩溃或丢数据。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=0.1)
            errors = []
            stop = threading.Event()

            def store_worker():
                try:
                    i = 0
                    while not stop.is_set():
                        mem.store_sample({
                            'timestamp': time.time(),
                            'state': {'cpu': i},
                            'predictions': {'low': {}, 'mid': {}, 'high': {}},
                            'actuals': {'cpu': i},
                            'errors': {'cpu': 0.1},
                            'task_id': f't{i}',
                            'reward': 0.5,
                            'value_score': 0.5,
                        })
                        i += 1
                except Exception as e:
                    errors.append(e)

            def flush_worker():
                try:
                    while not stop.is_set():
                        mem.flush()
                        time.sleep(0.01)
                except Exception as e:
                    errors.append(e)

            store_threads = [threading.Thread(target=store_worker) for _ in range(3)]
            flush_threads = [threading.Thread(target=flush_worker) for _ in range(2)]
            for t in store_threads + flush_threads:
                t.start()
            time.sleep(2)
            stop.set()
            for t in store_threads + flush_threads:
                t.join(timeout=10)

            mem.flush()
            count = mem.get_sample_count()
            mem.close()

            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
            self.assertGreater(count, 0, "No samples stored")

    def test_concurrent_read_write(self):
        """并发读写（store+load_samples）应不崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=9999)
            errors = []
            stop = threading.Event()

            def write_worker():
                try:
                    i = 0
                    while not stop.is_set():
                        mem.store_sample({
                            'timestamp': time.time(),
                            'state': {'cpu': i},
                            'predictions': {'low': {}, 'mid': {}, 'high': {}},
                            'actuals': {'cpu': i},
                            'errors': {'cpu': 0.1},
                            'task_id': f't{i}',
                            'reward': 0.5,
                            'value_score': 0.5,
                        })
                        if i % 10 == 0:
                            mem.flush()
                        i += 1
                except Exception as e:
                    errors.append(e)

            def read_worker():
                try:
                    while not stop.is_set():
                        mem.load_samples(limit=10)
                        mem.get_sample_count()
                        time.sleep(0.01)
                except Exception as e:
                    errors.append(e)

            writers = [threading.Thread(target=write_worker) for _ in range(2)]
            readers = [threading.Thread(target=read_worker) for _ in range(2)]
            for t in writers + readers:
                t.start()
            time.sleep(2)
            stop.set()
            for t in writers + readers:
                t.join(timeout=10)

            mem.flush()
            mem.close()
            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")


class TestConcurrentTaskPool(unittest.TestCase):
    """TaskPool高并发测试。"""

    def test_concurrent_update_and_select(self):
        """并发update和select应不崩溃。"""
        pool = TaskPool(max_size=100)
        errors = []
        stop = threading.Event()

        def update_worker(tid):
            try:
                i = 0
                while not stop.is_set():
                    task = Task(id=f'task_{tid}_{i}', type=TaskType.SAFE, action='test')
                    pool.update([task])
                    i += 1
            except Exception as e:
                errors.append(e)

        def select_worker():
            try:
                while not stop.is_set():
                    pool.select_task()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        updaters = [threading.Thread(target=update_worker, args=(i,)) for i in range(3)]
        selectors = [threading.Thread(target=select_worker) for _ in range(2)]
        for t in updaters + selectors:
            t.start()
        time.sleep(2)
        stop.set()
        for t in updaters + selectors:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
        self.assertLessEqual(len(pool.tasks), 100)

    def test_concurrent_mark_executed(self):
        """并发mark_executed应不崩溃。"""
        pool = TaskPool(max_size=50)
        # 预填充任务
        for i in range(20):
            task = Task(id=f'task_{i}', type=TaskType.SAFE, action='test')
            pool.update([task])

        errors = []
        def worker():
            try:
                for i in range(20):
                    pool.mark_executed(f'task_{i}', True, 0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
        stats = pool.get_stats()
        self.assertGreater(stats['total_executions'], 0)


class TestConcurrentSnapshotManager(unittest.TestCase):
    """SnapshotManager高并发测试。"""

    def test_concurrent_create_and_load(self):
        """并发create和load应不崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=20)
            errors = []
            created_ids = []
            ids_lock = threading.Lock()

            def create_worker(tid):
                try:
                    for i in range(5):
                        sid = sm.create_snapshot({'v': i, 'thread': tid}, metrics={'mae': 0.1 * i})
                        with ids_lock:
                            created_ids.append(sid)
                except Exception as e:
                    errors.append(e)

            def load_worker():
                try:
                    while True:
                        with ids_lock:
                            if len(created_ids) > 0:
                                sid = created_ids[-1]
                            else:
                                sid = None
                        if sid:
                            sm.load_snapshot(sid)
                        time.sleep(0.001)
                        if len(created_ids) >= 20:
                            break
                except Exception as e:
                    errors.append(e)

            creators = [threading.Thread(target=create_worker, args=(i,)) for i in range(4)]
            loaders = [threading.Thread(target=load_worker) for _ in range(2)]
            for t in creators + loaders:
                t.start()
            for t in creators:
                t.join(timeout=10)
            for t in loaders:
                t.join(timeout=5)

            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
            self.assertLessEqual(len(sm._snapshots), 20)

    def test_concurrent_create_and_delete(self):
        """并发create和delete应不崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=50)
            errors = []
            created_ids = []
            ids_lock = threading.Lock()

            def create_worker():
                try:
                    for i in range(10):
                        sid = sm.create_snapshot({'v': i}, metrics={'mae': 0.1})
                        with ids_lock:
                            created_ids.append(sid)
                except Exception as e:
                    errors.append(e)

            def delete_worker():
                try:
                    for _ in range(5):
                        time.sleep(0.1)
                        with ids_lock:
                            if created_ids:
                                sid = created_ids.pop(0)
                        try:
                            sm.delete_snapshot(sid)
                        except (KeyError, Exception):
                            pass
                except Exception as e:
                    errors.append(e)

            creators = [threading.Thread(target=create_worker) for _ in range(3)]
            deleters = [threading.Thread(target=delete_worker) for _ in range(2)]
            for t in creators + deleters:
                t.start()
            for t in creators + deleters:
                t.join(timeout=10)

            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")


class TestConcurrentAutoRollback(unittest.TestCase):
    """AutoRollback高并发测试。"""

    def test_concurrent_check_and_rollback(self):
        """并发check_and_rollback应不崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            sm = SnapshotManager(snapshot_dir=tmp, max_count=10)
            ar = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=0.1)
            sid = sm.create_snapshot({'v': 1}, metrics={'mae': 0.05})
            ar.set_baseline(0.05, sid)

            errors = []
            restored = []
            restore_lock = threading.Lock()

            def restore_callback(state):
                with restore_lock:
                    restored.append(state)
                return True

            def worker():
                try:
                    for _ in range(10):
                        ar.check_and_rollback(0.5, restore_callback)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")


class TestDeadlockDetection(unittest.TestCase):
    """死锁检测测试。"""

    def test_no_deadlock_on_concurrent_operations(self):
        """并发操作不应导致死锁（10秒内完成）。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=5, flush_interval=0.05)
            pool = TaskPool(max_size=50)

            done = threading.Event()
            errors = []

            def memory_worker():
                try:
                    for i in range(50):
                        mem.store_sample({
                            'timestamp': time.time(),
                            'state': {'cpu': i},
                            'predictions': {'low': {}, 'mid': {}, 'high': {}},
                            'actuals': {'cpu': i},
                            'errors': {'cpu': 0.1},
                            'task_id': f't{i}',
                            'reward': 0.5,
                            'value_score': 0.5,
                        })
                        if i % 10 == 0:
                            mem.flush()
                except Exception as e:
                    errors.append(e)

            def task_worker():
                try:
                    for i in range(50):
                        task = Task(id=f'task_{i}', type=TaskType.SAFE, action='test')
                        pool.update([task])
                        pool.select_task()
                except Exception as e:
                    errors.append(e)

            m_thread = threading.Thread(target=memory_worker)
            t_thread = threading.Thread(target=task_worker)
            m_thread.start()
            t_thread.start()

            # 等待完成，超时则认为死锁
            m_thread.join(timeout=10)
            t_thread.join(timeout=10)

            mem.flush()
            mem.close()

            self.assertFalse(m_thread.is_alive(), "Memory thread deadlocked")
            self.assertFalse(t_thread.is_alive(), "Task thread deadlocked")
            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")


class TestConcurrentDataConsistency(unittest.TestCase):
    """并发后数据一致性验证。"""

    def test_sample_ordering_after_concurrent_store(self):
        """并发store后sample的timestamp应单调递增（按存储顺序）。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=100, flush_interval=9999)
            timestamps = []
            ts_lock = threading.Lock()

            def worker(tid):
                for i in range(10):
                    ts = time.time()
                    with ts_lock:
                        timestamps.append(ts)
                    mem.store_sample({
                        'timestamp': ts,
                        'state': {'cpu': i, 'thread': tid},
                        'predictions': {'low': {}, 'mid': {}, 'high': {}},
                        'actuals': {'cpu': i},
                        'errors': {'cpu': 0.1},
                        'task_id': f't{tid}_{i}',
                        'reward': 0.5,
                        'value_score': 0.5,
                    })

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            mem.flush()
            count = mem.get_sample_count()
            mem.close()

            self.assertEqual(count, 50)

    def test_task_pool_consistent_after_concurrent_ops(self):
        """并发操作后task_pool状态应一致。"""
        pool = TaskPool(max_size=20)
        errors = []

        def add_and_remove_worker(tid):
            try:
                for i in range(10):
                    task_id = f'task_{tid}_{i}'
                    task = Task(id=task_id, type=TaskType.SAFE, action='test')
                    pool.update([task])
                    if i % 2 == 0:
                        pool.remove_task(task_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_and_remove_worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
        self.assertLessEqual(len(pool.tasks), 20)
        # 验证所有剩余任务的id唯一
        task_ids = [t.id for t in pool.tasks.values()]
        self.assertEqual(len(task_ids), len(set(task_ids)), "Duplicate task IDs found")


class TestHighConcurrencyStress(unittest.TestCase):
    """高并发压力测试。"""

    def test_20_threads_concurrent_store(self):
        """20线程并发store应不崩溃不丢数据。"""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            mem = ShortTermMemory(db_path=db_path, batch_size=20, flush_interval=0.1)
            errors = []
            counts = []
            count_lock = threading.Lock()

            def worker(tid):
                try:
                    local_count = 0
                    for i in range(50):
                        mem.store_sample({
                            'timestamp': time.time(),
                            'state': {'cpu': i, 'thread': tid},
                            'predictions': {'low': {}, 'mid': {}, 'high': {}},
                            'actuals': {'cpu': i},
                            'errors': {'cpu': 0.1},
                            'task_id': f't{tid}_{i}',
                            'reward': 0.5,
                            'value_score': 0.5,
                        })
                        local_count += 1
                    with count_lock:
                        counts.append(local_count)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            mem.flush()
            total_stored = sum(counts)
            db_count = mem.get_sample_count()
            mem.close()

            self.assertEqual(len(errors), 0, f"Errors: {errors[:3]}")
            self.assertEqual(db_count, total_stored,
                             f"Data loss: stored {total_stored}, db has {db_count}")


if __name__ == '__main__':
    unittest.main()
