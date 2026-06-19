#!/usr/bin/env python3
"""
长时间大规模压力测试：运行1000+周期，模拟各种复杂场景
目标：发现真实运行时的bug
"""

import os
import sys
import unittest
import tempfile
import shutil
import random
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config


class TestLongRunStress(unittest.TestCase):
    """长周期压力测试"""

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
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

    def test_100_cycles_basic(self):
        """基础100周期测试"""
        agent = AgentLoop('config.yaml')
        for i in range(100):
            result = agent.run_cycle()
            self.assertEqual(result['cycle'], i + 1)
            self.assertIn('predictions', result)
            self.assertIn('reward', result)
        agent.stop()

    def test_1000_cycles_stability(self):
        """1000周期稳定性测试"""
        agent = AgentLoop('config.yaml')
        errors_found = []

        for i in range(1000):
            try:
                result = agent.run_cycle()
                # 验证数据结构完整性
                self.assertIsInstance(result['cycle'], int)
                self.assertIsInstance(result['duration_seconds'], float)
                self.assertGreaterEqual(result['duration_seconds'], 0)

                # 验证奖励范围
                reward = result['reward']['total_reward']
                self.assertGreaterEqual(reward, 0.0)
                self.assertLessEqual(reward, 1.0)

            except Exception as e:
                errors_found.append((i + 1, str(e)))
                if len(errors_found) > 10:
                    break

        stats = agent.get_stats()
        agent.stop()

        if errors_found:
            self.fail(f"Errors in cycles: {errors_found[:5]}")

        self.assertEqual(stats['cycle_count'], 1000)

    def test_emergency_recovery_repeated(self):
        """反复触发紧急模式并恢复"""
        agent = AgentLoop('config.yaml')

        for _ in range(20):
            # 正常周期
            agent.run_cycle()

            # 触发紧急
            agent.resource_guardian.max_cpu_percent = 0.001
            agent.resource_guardian.max_mem_percent = 0.001
            result = agent.run_cycle()
            self.assertTrue(result['safety']['emergency_mode'])

            # 恢复
            agent.resource_guardian.max_cpu_percent = 100.0
            agent.resource_guardian.max_mem_percent = 100.0
            result = agent.run_cycle()
            self.assertFalse(result['safety']['emergency_mode'])

        agent.stop()

    def test_memory_pressure(self):
        """内存压力测试：大量数据写入"""
        agent = AgentLoop('config.yaml')

        for _ in range(300):
            agent.run_cycle()

        # 验证内存没有无限增长
        count = agent.memory.get_sample_count()
        self.assertLessEqual(count, 300 + 50)  # 允许一些缓冲

        agent.stop()

    def test_concurrent_access_simulation(self):
        """模拟并发访问场景"""
        agent = AgentLoop('config.yaml')
        results = []
        errors = []

        def worker():
            try:
                for _ in range(50):
                    r = agent.run_cycle()
                    results.append(r['cycle'])
            except Exception as e:
                errors.append(str(e))

        # 启动多个线程
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        agent.stop()

        # 不应该有重复cycle号（因为cycle_count是线程安全的）
        # 实际上由于GIL，这里可能不会有真正的并发问题
        # 但我们要确保没有崩溃
        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")

    def test_random_resource_fluctuation(self):
        """随机资源波动测试"""
        agent = AgentLoop('config.yaml')

        for _ in range(200):
            # 随机改变资源限制
            agent.resource_guardian.max_cpu_percent = random.uniform(0.001, 100.0)
            agent.resource_guardian.max_mem_percent = random.uniform(0.001, 100.0)

            result = agent.run_cycle()
            # 无论资源限制如何变化，都不应该崩溃
            self.assertIn('safety', result)

        agent.stop()

    def test_snapshot_at_boundaries(self):
        """在快照边界处测试"""
        agent = AgentLoop('config.yaml')

        # 运行到刚好触发快照的周期
        for i in range(105):
            result = agent.run_cycle()
            if (i + 1) % 100 == 0:
                self.assertIsNotNone(result['snapshot'])
                self.assertIn('path', result['snapshot'])

        agent.stop()

    def test_task_pool_under_pressure(self):
        """任务池高压测试"""
        agent = AgentLoop('config.yaml')

        for _ in range(200):
            agent.run_cycle()
            stats = agent.task_pool.get_stats()
            # 任务池不应该溢出或出现负数
            self.assertGreaterEqual(stats['total_tasks'], 0)
            self.assertGreaterEqual(stats['total_executions'], 0)

        agent.stop()

    def test_reward_consistency_under_load(self):
        """高负载下奖励一致性"""
        agent = AgentLoop('config.yaml')
        rewards = []

        for _ in range(500):
            result = agent.run_cycle()
            reward = result['reward']['total_reward']
            rewards.append(reward)

            # 验证safe/gain/eff各自也在合理范围
            self.assertGreaterEqual(result['reward']['safe_reward'], 0.0)
            self.assertGreaterEqual(result['reward']['gain_reward'], 0.0)
            self.assertGreaterEqual(result['reward']['eff_reward'], 0.0)

        agent.stop()

        # 奖励分布应该合理
        avg_reward = sum(rewards) / len(rewards)
        self.assertGreater(avg_reward, 0.0)
        self.assertLess(avg_reward, 1.0)

    def test_prediction_drift_over_time(self):
        """长时间运行后预测器漂移检测"""
        agent = AgentLoop('config.yaml')

        mae_history = []
        for _ in range(200):
            result = agent.run_cycle()
            mae = sum(result['errors'].values()) / max(len(result['errors']), 1)
            mae_history.append(mae)

        agent.stop()

        # MAE不应该无限增长（预测器应该自我修正）
        early_mae = sum(mae_history[:20]) / 20
        late_mae = sum(mae_history[-20:]) / 20

        # 允许一定增长，但不应该爆炸性增长
        self.assertLess(late_mae, early_mae * 5 + 1.0)

    def test_full_loop_with_all_features(self):
        """全功能综合测试"""
        agent = AgentLoop('config.yaml')

        for i in range(150):
            result = agent.run_cycle()

            # 每50周期检查一次统计数据
            if (i + 1) % 50 == 0:
                stats = agent.get_stats()
                self.assertEqual(stats['cycle_count'], i + 1)
                self.assertGreaterEqual(stats['memory_samples'], 0)
                self.assertGreaterEqual(stats['recent_mae'], 0.0)
                self.assertGreaterEqual(stats['recent_success_rate'], 0.0)
                self.assertLessEqual(stats['recent_success_rate'], 1.0)

        agent.stop()


if __name__ == '__main__':
    unittest.main()
