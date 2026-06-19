#!/usr/bin/env python3
"""
超大规模混沌实战测试
目标：1000+周期运行，多异常并发注入，验证系统极限稳定性
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config


class TestMassiveChaos(unittest.TestCase):
    """超大规模混沌实战测试"""

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
  min_samples: 99999
memory:
  max_short_term_samples: 500
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

    def test_300_cycles_with_random_faults(self):
        """300个周期随机故障注入"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(300):
            try:
                if i % 100 == 0:
                    with patch.object(agent.low_predictor, 'predict', side_effect=RuntimeError('predictor fault')):
                        agent.run_cycle()
                elif i % 77 == 0:
                    with patch.object(agent.memory, 'store_sample', side_effect=Exception('memory fault')):
                        agent.run_cycle()
                elif i % 53 == 0:
                    with patch.object(agent.reward_engine, 'compute_total_reward', side_effect=ValueError('reward fault')):
                        agent.run_cycle()
                elif i % 31 == 0:
                    with patch.object(agent.safety_enforcer, 'full_check', side_effect=OSError('safety fault')):
                        agent.run_cycle()
                else:
                    agent.run_cycle()
            except Exception as e:
                errors.append(f"cycle_{i}: {e}")

        agent.stop()
        self.assertEqual(len(errors), 0, f"Massive chaos test failed: {errors}")
        self.assertGreaterEqual(agent._cycle_count, 300)

    def test_concurrent_access_during_cycles(self):
        """多线程并发访问Agent状态"""
        agent = AgentLoop('config.yaml')
        errors = []
        lock = threading.Lock()

        def cycle_worker():
            for _ in range(50):
                try:
                    agent.run_cycle()
                except Exception as e:
                    with lock:
                        errors.append(str(e))

        def stats_worker():
            for _ in range(100):
                try:
                    stats = agent.get_stats()
                    self.assertIn('cycle_count', stats)
                except Exception as e:
                    with lock:
                        errors.append(str(e))
                time.sleep(0.001)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=cycle_worker))
        for _ in range(2):
            threads.append(threading.Thread(target=stats_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        agent.stop()
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")

    def test_rapid_restart_survival(self):
        """快速重启Agent多次仍能存活"""
        errors = []
        for i in range(10):
            try:
                Config._instance = None
                Config._config = {}
                agent = AgentLoop('config.yaml')
                for _ in range(20):
                    agent.run_cycle()
                agent.stop()
            except Exception as e:
                errors.append(f"restart_{i}: {e}")
        self.assertEqual(len(errors), 0, f"Rapid restart failed: {errors}")

    def test_mixed_exception_storm(self):
        """混合异常风暴：同一周期内多个组件同时异常"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(100):
            try:
                with patch.object(agent.low_predictor, 'predict', side_effect=RuntimeError('storm')):
                    with patch.object(agent.reward_engine, 'compute_total_reward', side_effect=ValueError('storm')):
                        with patch.object(agent.memory, 'store_sample', side_effect=Exception('storm')):
                            agent.run_cycle()
            except Exception as e:
                errors.append(f"storm_cycle_{i}: {e}")

        agent.stop()
        self.assertEqual(len(errors), 0, f"Exception storm crashed agent: {errors}")

    def test_memory_pressure_with_high_frequency(self):
        """高频运行下的内存压力测试"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(200):
            try:
                agent.run_cycle()
            except Exception as e:
                errors.append(f"pressure_cycle_{i}: {e}")

        agent.stop()
        self.assertEqual(len(errors), 0, f"Memory pressure test failed: {errors}")
        stats = agent.get_stats()
        self.assertLessEqual(stats['memory_samples'], 500)


if __name__ == '__main__':
    unittest.main()
