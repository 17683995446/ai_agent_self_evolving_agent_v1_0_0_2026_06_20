#!/usr/bin/env python3
"""
长时运行混沌测试
目标：模拟500+周期长时间运行，在复杂场景下发现系统真实bug
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
from src.task.task_definition import Task, TaskType


class TestLongRunningChaos(unittest.TestCase):
    """长时运行混沌测试"""

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
  coverage_evolution: true
  efficiency_evolution: true
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: 100.0
  max_mem_percent: 100.0
  max_db_size_mb: 100
  max_snapshot_count: 5
exploration:
  base_ratio: 0.3
  stable_threshold: 0.05
  emergency_ratio: 0.1
  novel_ratio: 0.4
safety:
  max_consecutive_failures: 3
  forbidden_paths:
    - /boot
    - /etc
review:
  interval_hours: 0.001
  min_samples: 5
  idle_cpu_threshold: 100.0
  idle_duration_seconds: 1
memory:
  max_short_term_samples: 300
  value_retention_ratio: 0.3
  history_window: 50
logging:
  level: INFO
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

    def test_200_cycles_with_periodic_faults(self):
        """200周期运行，每50周期注入一次故障"""
        agent = AgentLoop('config.yaml')
        errors = []
        results = []

        try:
            for i in range(200):
                result = agent.run_cycle()
                results.append(result)

                # 每50周期注入故障
                if i > 0 and i % 50 == 0:
                    fault_type = random.choice([
                        'extreme_cpu_limit',
                        'corrupt_db',
                        'freeze_all_tasks',
                        'clear_state_history',
                        'rapid_config_change',
                    ])
                    if fault_type == 'extreme_cpu_limit':
                        agent.resource_guardian.max_cpu_percent = 0.001
                    elif fault_type == 'corrupt_db':
                        db_path = os.path.join(self.temp_dir, 'data', 'memory.db')
                        if os.path.exists(db_path):
                            with open(db_path, 'wb') as f:
                                f.write(b'CORRUPT')
                    elif fault_type == 'freeze_all_tasks':
                        for tid in list(agent.task_pool.tasks.keys()):
                            agent.task_pool.freeze_task(tid)
                    elif fault_type == 'clear_state_history':
                        agent.state_vector._recent_mae_history.clear()
                        agent.state_vector._recent_task_success.clear()
                    elif fault_type == 'rapid_config_change':
                        agent.config.set('exploration.base_ratio', random.random())

                # 每100周期恢复资源限制
                if i > 0 and i % 100 == 0:
                    agent.resource_guardian.max_cpu_percent = 100.0

        except Exception as e:
            errors.append(f"cycle_{i}: {e}")

        agent.stop()

        # 验证
        self.assertEqual(len(errors), 0, f"Long run errors: {errors}")
        self.assertEqual(len(results), 200)
        self.assertTrue(all('cycle' in r for r in results))

    def test_100_cycles_with_concurrent_dashboard(self):
        """100周期运行，同时启动仪表盘并并发访问"""
        agent = AgentLoop('config.yaml')
        errors = []

        # 启动仪表盘
        try:
            agent.dashboard.start()
        except Exception:
            pass

        def cycle_runner():
            try:
                for _ in range(100):
                    agent.run_cycle()
            except Exception as e:
                errors.append(f"runner: {e}")

        def dashboard_poller():
            try:
                import urllib.request
                for _ in range(25):
                    time.sleep(0.5)
                    try:
                        urllib.request.urlopen(
                            f'http://{agent.dashboard.host}:{agent.dashboard.port}/api/data',
                            timeout=1
                        )
                    except Exception:
                        pass
            except Exception as e:
                errors.append(f"poller: {e}")

        t1 = threading.Thread(target=cycle_runner)
        t2 = threading.Thread(target=dashboard_poller)
        t1.start()
        t2.start()
        t1.join(timeout=120)
        t2.join(timeout=120)

        try:
            agent.dashboard.stop()
        except Exception:
            pass
        agent.stop()

        self.assertEqual(len(errors), 0, f"Dashboard concurrent errors: {errors}")

    def test_state_vector_under_load(self):
        """状态向量在负载下的稳定性"""
        agent = AgentLoop('config.yaml')
        state_history = []

        for i in range(150):
            agent.run_cycle()
            state_history.append(agent.state_vector.flatten(agent.state_vector.perceive()))

            # 强制清空状态历史
            if i == 50:
                agent.state_vector._recent_mae_history.clear()
                agent.state_vector._recent_task_success.clear()
            elif i == 100:
                agent.state_vector._recent_mae_history.clear()

        agent.stop()

        # 验证状态记录完整
        self.assertEqual(len(state_history), 150)
        self.assertTrue(all(isinstance(s, dict) for s in state_history))

    def test_memory_growth_under_sustained_load(self):
        """持续负载下的内存增长检测"""
        import psutil
        agent = AgentLoop('config.yaml')
        process = psutil.Process()

        mem_samples = []
        for i in range(100):
            agent.run_cycle()
            if i % 20 == 0:
                mem_mb = process.memory_info().rss / (1024 ** 2)
                mem_samples.append(mem_mb)

        agent.stop()

        # 内存增长不应超过50MB
        if len(mem_samples) >= 2:
            growth = mem_samples[-1] - mem_samples[0]
            self.assertLess(growth, 50.0, f"Memory growth too high: {growth:.1f}MB")

    def test_rapid_task_generation_and_execution(self):
        """快速任务生成与执行压力测试"""
        agent = AgentLoop('config.yaml')

        # 预生成大量任务
        for i in range(100):
            agent.task_pool.update([
                Task(id=f'load_{i}', type=TaskType.EXPLORE, name=f'load_task_{i}')
            ])

        errors = []
        for i in range(50):
            try:
                agent.run_cycle()
            except Exception as e:
                errors.append(str(e))

        agent.stop()

        stats = agent.get_stats()
        self.assertEqual(len(errors), 0, f"Task load errors: {errors}")
        self.assertLessEqual(stats['task_pool']['total_tasks'], 50)


if __name__ == '__main__':
    unittest.main()
