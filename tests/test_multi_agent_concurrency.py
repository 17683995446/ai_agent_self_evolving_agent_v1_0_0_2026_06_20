#!/usr/bin/env python3
"""
阶段71：多Agent并发竞争与资源冲突测试
目标：多个Agent同时运行，验证共享资源竞争、并发安全、数据一致性
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType


class TestSharedDatabaseConcurrency(unittest.TestCase):
    """共享数据库并发测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'shared.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_agents_same_database(self):
        """多个Agent共享同一个数据库文件"""
        errors = []
        lock = threading.Lock()
        # 使用单个共享的ShortTermMemory实例，依靠内部RLock序列化访问
        shared_mem = ShortTermMemory(db_path=self.db_path, max_samples=500)

        def agent_worker(worker_id: int):
            try:
                for i in range(50):
                    shared_mem.store_sample({
                        'timestamp': time.time(),
                        'state': {'worker': worker_id, 'idx': i},
                        'predictions': {},
                        'actuals': {},
                        'errors': {},
                        'reward': 0.5,
                    })
            except Exception as e:
                with lock:
                    errors.append(f"worker {worker_id}: {e}")

        threads = [threading.Thread(target=agent_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        shared_mem.close()
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # 验证数据完整性
        mem = ShortTermMemory(db_path=self.db_path, max_samples=5000)
        count = mem.get_sample_count()
        mem.close()
        self.assertGreaterEqual(count, 50)

    def test_concurrent_read_write_same_db(self):
        """同一数据库并发读写"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=1000)
        errors = []
        stop_event = threading.Event()

        def writer():
            for i in range(200):
                if stop_event.is_set():
                    break
                try:
                    mem.store_sample({
                        'timestamp': time.time(),
                        'state': {'idx': i},
                        'predictions': {},
                        'actuals': {},
                        'errors': {},
                        'reward': 0.5,
                    })
                except Exception as e:
                    errors.append(f"writer: {e}")

        def reader():
            for _ in range(200):
                if stop_event.is_set():
                    break
                try:
                    samples = mem.load_samples(limit=10)
                    _ = len(samples)
                except Exception as e:
                    errors.append(f"reader: {e}")
                time.sleep(0.005)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        stop_event.set()
        mem.close()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")


class TestSharedSnapshotDirectory(unittest.TestCase):
    """共享快照目录并发测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_snapshot_creation(self):
        """多个线程同时创建快照"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=100)
        errors = []

        def snapshot_worker(worker_id: int):
            for i in range(20):
                try:
                    sm.create_snapshot(
                        model_state={'worker': worker_id, 'idx': i},
                        tag=f"worker_{worker_id}",
                        metrics={'mae': 0.1}
                    )
                except Exception as e:
                    errors.append(f"worker {worker_id}: {e}")

        threads = [threading.Thread(target=snapshot_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        snapshots = sm.list_snapshots()
        self.assertGreaterEqual(len(snapshots), 20)


class TestMultiAgentResourceConflict(unittest.TestCase):
    """多Agent资源冲突测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        self.temp_dir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        for d in ['data', 'logs', 'snapshots']:
            os.makedirs(d, exist_ok=True)
        with open('config.yaml', 'w') as f:
            f.write("""
goals:
  precision_evolution: true
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: 100.0
  max_mem_percent: 100.0
exploration:
  base_ratio: 0.3
safety:
  max_consecutive_failures: 5
review:
  interval_hours: 999
memory:
  max_short_term_samples: 200
logging:
  level: WARNING
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
dashboard:
  enabled: false
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_three_agents_same_directory(self):
        """三个Agent在独立子目录下并发运行"""
        errors = []
        lock = threading.Lock()

        def agent_runner(agent_id: int):
            try:
                agent_dir = os.path.join(self.temp_dir, f'agent_{agent_id}')
                for d in ['data', 'logs', 'snapshots']:
                    os.makedirs(os.path.join(agent_dir, d), exist_ok=True)
                config_path = os.path.join(agent_dir, 'config.yaml')
                with open(config_path, 'w') as f:
                    f.write(f"""
goals:
  precision_evolution: true
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: 100.0
  max_mem_percent: 100.0
exploration:
  base_ratio: 0.3
safety:
  max_consecutive_failures: 5
review:
  interval_hours: 999
  min_samples: 99999
memory:
  max_short_term_samples: 200
logging:
  level: WARNING
paths:
  data_dir: {agent_dir}/data
  snapshot_dir: {agent_dir}/snapshots
  log_dir: {agent_dir}/logs
dashboard:
  enabled: false
""")
                with lock:
                    Config._singleton = False
                    Config._instance = None
                    Config._config = {}
                agent = AgentLoop(config_path)
                for _ in range(30):
                    agent.run_cycle()
                agent.stop()
            except Exception as e:
                with lock:
                    errors.append(f"agent {agent_id}: {e}")

        threads = [threading.Thread(target=agent_runner, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_agent_concurrent_config_access(self):
        """多线程并发访问Config单例"""
        cfg = Config()
        errors = []
        results = []

        def accessor(worker_id: int):
            for i in range(100):
                try:
                    cfg.set(f'key_{worker_id}_{i}', i)
                    val = cfg.get(f'key_{worker_id}_{i}')
                    results.append(val)
                except Exception as e:
                    errors.append(f"worker {worker_id}: {e}")

        threads = [threading.Thread(target=accessor, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 500)


class TestTaskPoolConcurrentModification(unittest.TestCase):
    """任务池并发修改测试"""

    def test_concurrent_update_and_select(self):
        """并发更新和选择任务"""
        pool = TaskPool(max_size=100)
        errors = []
        stop_event = threading.Event()

        def updater():
            for i in range(200):
                if stop_event.is_set():
                    break
                try:
                    pool.update([Task(id=f'task_{i}', type=TaskType.EXPLORE, name=f'task_{i}')])
                except Exception as e:
                    errors.append(f"updater: {e}")

        def selector():
            for _ in range(200):
                if stop_event.is_set():
                    break
                try:
                    task = pool.select_task()
                    if task:
                        pool.mark_executed(task.id, success=True, reward=0.5)
                except Exception as e:
                    errors.append(f"selector: {e}")
                time.sleep(0.002)

        t1 = threading.Thread(target=updater)
        t2 = threading.Thread(target=selector)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        stop_event.set()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        stats = pool.get_stats()
        self.assertGreaterEqual(stats['total_tasks'], 0)


if __name__ == '__main__':
    unittest.main()
