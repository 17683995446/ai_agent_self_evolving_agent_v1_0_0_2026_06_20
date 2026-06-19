#!/usr/bin/env python3
"""
复杂并发与资源竞争场景测试
目标：验证多线程高并发下无崩溃、无数据错乱、无死锁
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType
from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.safety.guardian import ResourceGuardian, BehaviorGuardian, SafetyEnforcer


class TestAgentLoopConcurrent(unittest.TestCase):
    """AgentLoop多线程并发测试"""

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
  interval_hours: 0.001
  min_samples: 10
memory:
  max_short_term_samples: 200
logging:
  level: DEBUG
  file: logs/agent.log
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

    def test_concurrent_run_cycle_multiple_threads(self):
        """多线程同时执行run_cycle"""
        agent = AgentLoop('config.yaml')
        errors = []
        cycles = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(30):
                    r = agent.run_cycle()
                    with lock:
                        cycles.append(r['cycle'])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        agent.stop()
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(cycles), 150)
        # cycle号应该唯一（无重复计数）
        self.assertEqual(len(set(cycles)), 150)

    def test_concurrent_run_cycle_with_stop(self):
        """运行时突然stop不崩溃"""
        agent = AgentLoop('config.yaml')
        errors = []

        def runner():
            try:
                for _ in range(100):
                    agent.run_cycle()
            except Exception as e:
                errors.append(str(e))

        def stopper():
            time.sleep(0.05)
            agent.stop()

        t1 = threading.Thread(target=runner)
        t2 = threading.Thread(target=stopper)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_concurrent_mixed_operations(self):
        """混合操作并发：run_cycle + get_stats + 修改配置"""
        agent = AgentLoop('config.yaml')
        errors = []

        def worker_a():
            try:
                for _ in range(20):
                    agent.run_cycle()
            except Exception as e:
                errors.append(f"A: {e}")

        def worker_b():
            try:
                for _ in range(20):
                    agent.get_stats()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"B: {e}")

        def worker_c():
            try:
                for _ in range(20):
                    agent.resource_guardian.max_cpu_percent = random.uniform(0.001, 100.0)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"C: {e}")

        threads = [
            threading.Thread(target=worker_a),
            threading.Thread(target=worker_b),
            threading.Thread(target=worker_c),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        agent.stop()
        self.assertEqual(len(errors), 0, f"Errors: {errors}")


class TestTaskPoolConcurrent(unittest.TestCase):
    """TaskPool并发竞争测试"""

    def test_concurrent_update_and_select(self):
        pool = TaskPool(max_size=50)
        errors = []

        def updater():
            for i in range(100):
                task = Task(id=f't_{i}', type=TaskType.EXPLORE, name=f'task_{i}')
                pool.update([task])

        def selector():
            for _ in range(100):
                pool.select_task()

        def scorer():
            for _ in range(100):
                pool.score_all({}, {'safe_gap': 0.1, 'gain_gap': 0.1, 'eff_gap': 0.1})

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=selector),
            threading.Thread(target=scorer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = pool.get_stats()
        self.assertGreaterEqual(stats['total_tasks'], 0)
        self.assertLessEqual(stats['total_tasks'], 50)

    def test_concurrent_mark_executed_and_freeze(self):
        pool = TaskPool(max_size=20)
        for i in range(10):
            pool.update([Task(id=f't_{i}', type=TaskType.EXPLORE, name=f'task_{i}')])

        errors = []

        def executer():
            for i in range(50):
                tid = f't_{i % 10}'
                pool.mark_executed(tid, success=(i % 2 == 0), reward=0.5)

        def freezer():
            for i in range(50):
                tid = f't_{i % 10}'
                pool.freeze_task(tid)
                pool.unfreeze_task(tid)

        threads = [
            threading.Thread(target=executer),
            threading.Thread(target=freezer),
            threading.Thread(target=executer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = pool.get_stats()
        self.assertGreaterEqual(stats['total_executions'], 0)


class TestShortTermMemoryConcurrent(unittest.TestCase):
    """SQLite并发压力测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'concurrent.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_high_concurrency_store_and_load(self):
        mem = ShortTermMemory(db_path=self.db_path, max_samples=5000)
        errors = []

        def writer():
            for i in range(100):
                try:
                    mem.store_sample({'reward': float(i), 'value_score': random.random()})
                except Exception as e:
                    errors.append(f"write: {e}")

        def reader():
            for _ in range(100):
                try:
                    mem.load_samples(limit=10)
                    mem.get_sample_count()
                except Exception as e:
                    errors.append(f"read: {e}")

        def cleaner():
            for _ in range(20):
                try:
                    mem.cleanup_old_samples(retention_ratio=0.5)
                except Exception as e:
                    errors.append(f"clean: {e}")

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=cleaner))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        count = mem.get_sample_count()
        self.assertGreaterEqual(count, 0)
        self.assertLessEqual(count, 5000)


class TestSnapshotManagerConcurrent(unittest.TestCase):
    """快照管理器并发测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_create_and_list(self):
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=20)
        errors = []

        def creator():
            for i in range(30):
                try:
                    sm.create_snapshot(
                        model_state={'cycle': i},
                        metrics={'mae': 0.1}
                    )
                except Exception as e:
                    errors.append(str(e))

        def lister():
            for _ in range(30):
                try:
                    sm.list_snapshots()
                except Exception as e:
                    errors.append(str(e))

        threads = [
            threading.Thread(target=creator),
            threading.Thread(target=creator),
            threading.Thread(target=lister),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertLessEqual(len(sm.list_snapshots()), 20)


class TestSafetyEnforcerConcurrent(unittest.TestCase):
    """安全模块并发测试"""

    def test_concurrent_resource_check(self):
        rg = ResourceGuardian(max_cpu_percent=100.0)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        errors = []

        def checker():
            for _ in range(100):
                try:
                    se.full_check()
                    se.is_emergency()
                except Exception as e:
                    errors.append(str(e))

        def modifier():
            for _ in range(100):
                rg.max_cpu_percent = random.uniform(0.001, 200.0)

        threads = [
            threading.Thread(target=checker),
            threading.Thread(target=checker),
            threading.Thread(target=modifier),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")


if __name__ == '__main__':
    unittest.main()
