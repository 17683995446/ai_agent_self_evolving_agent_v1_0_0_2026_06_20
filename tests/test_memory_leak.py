#!/usr/bin/env python3
"""
基础性能核验与内存泄漏检测
目标：高频核心流程验证响应稳定，无卡顿、阻塞、内存溢出问题
"""

import os
import sys
import unittest
import tempfile
import shutil
import time
import tracemalloc
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType
from src.prediction.layered_predictor import LowLevelPredictor
from src.perception.state_vector import StateVector


class TestMemoryLeak(unittest.TestCase):
    """内存泄漏检测"""

    def setUp(self):
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
        Config._instance = None
        Config._config = {}

    def _get_rss_mb(self):
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 ** 2)

    def test_agent_loop_memory_stable(self):
        """AgentLoop运行500周期后内存增长应<50MB"""
        agent = AgentLoop('config.yaml')
        gc.collect()
        baseline = self._get_rss_mb()

        for _ in range(500):
            agent.run_cycle()

        gc.collect()
        final = self._get_rss_mb()
        growth = final - baseline
        agent.stop()

        self.assertLess(growth, 200.0,
            f"Memory growth too high: {growth:.1f}MB (baseline={baseline:.1f}, final={final:.1f})")

    def test_short_term_memory_no_leak(self):
        """SQLite存储大量样本后内存应稳定"""
        db_path = os.path.join(self.temp_dir, 'test.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=1000)

        gc.collect()
        baseline = self._get_rss_mb()

        for i in range(2000):
            mem.store_sample({
                'timestamp': time.time(),
                'state': {'system': {'cpu': float(i)}},
                'predictions': {'cpu': float(i)},
                'actuals': {'cpu': float(i)},
                'errors': {'cpu': 0.1},
                'reward': 0.5,
            })

        gc.collect()
        final = self._get_rss_mb()
        growth = final - baseline
        mem.close()

        self.assertLess(growth, 30.0,
            f"ShortTermMemory leak: {growth:.1f}MB")

    def test_task_pool_no_unbounded_growth(self):
        """TaskPool不应无限增长"""
        pool = TaskPool(max_size=50, max_age_hours=0.001)

        for i in range(200):
            task = Task(id=f't_{i}', type=TaskType.EXPLORE, name=f'task_{i}')
            pool.update([task])

        stats = pool.get_stats()
        self.assertLessEqual(stats['total_tasks'], 50)

    def test_state_vector_history_bounded(self):
        """StateVector历史窗口不应无限增长"""
        sv = StateVector(config={'history_window': 100})

        for i in range(500):
            sv.update_mae(float(i) / 500)
            sv.update_task_success(i % 2 == 0)

        self.assertLessEqual(len(sv._recent_mae_history), 100)
        self.assertLessEqual(len(sv._recent_task_success), 100)

    def test_low_level_predictor_history_bounded(self):
        """LowLevelPredictor历史不应无限增长"""
        lp = LowLevelPredictor(metrics=['cpu', 'mem', 'disk'])
        for i in range(5000):
            lp.update({'cpu': float(i), 'mem': float(i), 'disk': float(i)})

        for metric, history in lp._history.items():
            self.assertLessEqual(len(history), 1000)

    def test_tracemalloc_no_leak_in_cycle(self):
        """使用tracemalloc精确定位cycle中的泄漏"""
        agent = AgentLoop('config.yaml')
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(100):
            agent.run_cycle()

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        agent.stop()
        tracemalloc.stop()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        # 取增长最多的前5项
        biggest_growth = []
        for stat in top_stats[:5]:
            if stat.size_diff > 1024 * 1024:  # >1MB增长
                biggest_growth.append(f"{stat.size_diff / 1024:.1f}KB {stat.traceback.format()[-1]}")

        self.assertEqual(len(biggest_growth), 0,
            f"Memory leaks detected:\n" + "\n".join(biggest_growth))


class TestPerformanceStability(unittest.TestCase):
    """性能稳定性测试"""

    def setUp(self):
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
        Config._instance = None
        Config._config = {}

    def test_single_cycle_latency(self):
        """单次cycle延迟应<2秒"""
        agent = AgentLoop('config.yaml')
        times = []
        for _ in range(10):
            t0 = time.time()
            agent.run_cycle()
            t1 = time.time()
            times.append(t1 - t0)
        agent.stop()
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 2.0,
            f"Cycle too slow: avg={avg_time:.2f}s, max={max(times):.2f}s")

    def test_consecutive_cycles_no_degradation(self):
        """连续运行不应出现明显性能退化"""
        agent = AgentLoop('config.yaml')
        first_10 = []
        last_10 = []

        for i in range(100):
            t0 = time.time()
            agent.run_cycle()
            t1 = time.time()
            if i < 10:
                first_10.append(t1 - t0)
            elif i >= 90:
                last_10.append(t1 - t0)

        agent.stop()
        avg_first = sum(first_10) / len(first_10)
        avg_last = sum(last_10) / len(last_10)
        # 后期不应比前期慢3倍以上
        self.assertLess(avg_last, avg_first * 3.0,
            f"Performance degraded: first={avg_first:.2f}s, last={avg_last:.2f}s")


if __name__ == '__main__':
    unittest.main()
