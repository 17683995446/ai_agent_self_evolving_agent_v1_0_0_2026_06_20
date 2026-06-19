#!/usr/bin/env python3
"""
阶段68：全链路压力测试与性能基线建立
目标：高并发/高负载下验证系统稳定性，建立各模块性能基线
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time
import gc
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType


class TestHighVolumeCycles(unittest.TestCase):
    """高容量周期压力测试"""

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
  level: WARNING
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

    def test_1000_cycles_no_crash(self):
        """连续运行1000个周期不崩溃"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(1000):
            try:
                agent.run_cycle()
            except Exception as e:
                errors.append(f"cycle {i}: {e}")
                if len(errors) > 5:
                    break

        agent.stop()
        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_cycle_duration_baseline(self):
        """测量单次周期耗时基线"""
        agent = AgentLoop('config.yaml')
        durations = []

        # 预热10个周期
        for _ in range(10):
            agent.run_cycle()

        # 测量100个周期
        for _ in range(100):
            t0 = time.perf_counter()
            agent.run_cycle()
            t1 = time.perf_counter()
            durations.append(t1 - t0)

        agent.stop()

        avg = sum(durations) / len(durations)
        p95 = sorted(durations)[int(len(durations) * 0.95)]
        p99 = sorted(durations)[int(len(durations) * 0.99)]

        # 基线：平均应小于500ms，P95小于1秒
        self.assertLess(avg, 0.5, f"Average cycle too slow: {avg:.3f}s")
        self.assertLess(p95, 1.0, f"P95 cycle too slow: {p95:.3f}s")
        self.assertLess(p99, 2.0, f"P99 cycle too slow: {p99:.3f}s")

    def test_memory_no_leak_over_500_cycles(self):
        """500个周期后内存增长应可控"""
        agent = AgentLoop('config.yaml')
        process = psutil.Process(os.getpid())

        # 强制GC后记录基线
        gc.collect()
        mem_before = process.memory_info().rss / (1024 ** 2)

        for _ in range(500):
            agent.run_cycle()

        gc.collect()
        mem_after = process.memory_info().rss / (1024 ** 2)
        agent.stop()

        growth_mb = mem_after - mem_before
        # 允许增长不超过50MB（SQLite和任务池正常增长）
        self.assertLess(growth_mb, 50.0, f"Memory leak suspected: {growth_mb:.1f}MB growth")

    def test_concurrent_10_threads_100_cycles_each(self):
        """10个线程各运行100个周期，每个线程使用独立数据目录避免DB锁竞争"""
        from src.core.config import Config
        errors = []
        lock = threading.Lock()

        def worker(worker_id: int):
            try:
                worker_dir = os.path.join(self.temp_dir, f'worker_{worker_id}')
                for d in ['data', 'logs', 'snapshots']:
                    os.makedirs(os.path.join(worker_dir, d), exist_ok=True)
                config_path = os.path.join(worker_dir, 'config.yaml')
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
  data_dir: {worker_dir}/data
  snapshot_dir: {worker_dir}/snapshots
  log_dir: {worker_dir}/logs
dashboard:
  enabled: false
""")
                with lock:
                    Config._singleton = False
                    Config._instance = None
                    Config._config = {}
                agent = AgentLoop(config_path)
                for _ in range(100):
                    agent.run_cycle()
                agent.stop()
            except Exception as e:
                with lock:
                    errors.append(f"worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")


class TestModulePerformanceBaseline(unittest.TestCase):
    """各模块性能基线测试"""

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

    def test_perceive_performance(self):
        """感知层性能基线"""
        agent = AgentLoop('config.yaml')
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            agent.state_vector.perceive()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        agent.stop()
        avg = sum(times) / len(times)
        self.assertLess(avg, 0.1, f"Perceive too slow: {avg:.3f}s")

    def test_predict_performance(self):
        """预测层性能基线"""
        agent = AgentLoop('config.yaml')
        # 预热更新一些数据
        agent.low_predictor.update({'cpu_percent': 10.0, 'mem_percent': 20.0, 'disk_percent': 30.0})
        agent.mid_predictor.update({'cpu_percent': 10.0, 'mem_percent': 20.0, 'disk_percent': 30.0})
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            low = agent.low_predictor.predict()
            mid = agent.mid_predictor.predict()
            agent.high_predictor.predict(low, mid)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        agent.stop()
        avg = sum(times) / len(times)
        self.assertLess(avg, 0.2, f"Predict too slow: {avg:.3f}s")

    def test_task_select_performance(self):
        """任务选择性能基线"""
        agent = AgentLoop('config.yaml')
        # 填充任务池
        for i in range(100):
            agent.task_pool.update([Task(id=f'perf_{i}', type=TaskType.EXPLORE, name=f'task_{i}')])
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            agent.task_pool.select_task()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        agent.stop()
        avg = sum(times) / len(times)
        self.assertLess(avg, 0.01, f"Task select too slow: {avg:.3f}s")

    def test_memory_store_performance(self):
        """记忆存储性能基线"""
        agent = AgentLoop('config.yaml')
        times = []
        for i in range(200):
            t0 = time.perf_counter()
            agent.memory.store_sample({
                'timestamp': time.time(),
                'state': {'cpu': i},
                'predictions': {'cpu': i + 1},
                'actuals': {'cpu': i},
                'errors': {'cpu': 1.0},
                'reward': 0.5,
            })
            t1 = time.perf_counter()
            times.append(t1 - t0)
        agent.stop()
        avg = sum(times) / len(times)
        self.assertLess(avg, 0.05, f"Memory store too slow: {avg:.3f}s")


class TestStressTaskPoolAndMemory(unittest.TestCase):
    """任务池与记忆系统高压测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'stress.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_task_pool_1000_tasks(self):
        """任务池处理1000个任务"""
        pool = TaskPool(max_size=1000)
        for i in range(1000):
            pool.update([Task(id=f'stress_{i}', type=TaskType.EXPLORE, name=f'task_{i}')])

        stats = pool.get_stats()
        self.assertLessEqual(stats['total_tasks'], 1000)

        # 选择任务不应崩溃
        for _ in range(100):
            task = pool.select_task()
            if task:
                pool.mark_executed(task.id, success=True, reward=0.5)

    def test_short_term_memory_10000_samples(self):
        """短期记忆存储10000条样本后手动清理"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=5000)
        errors = []

        for i in range(10000):
            try:
                mem.store_sample({
                    'timestamp': time.time(),
                    'state': {'idx': i},
                    'predictions': {'idx': i},
                    'actuals': {'idx': i},
                    'errors': {'idx': 0.1},
                    'reward': 0.5,
                })
            except Exception as e:
                errors.append(str(e))

        # 手动触发清理（模拟review_engine的行为）
        mem.cleanup_old_samples(retention_ratio=0.3)
        count = mem.get_sample_count()
        mem.close()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # 清理后应不超过max_samples
        self.assertLessEqual(count, 5000)


if __name__ == '__main__':
    unittest.main()
