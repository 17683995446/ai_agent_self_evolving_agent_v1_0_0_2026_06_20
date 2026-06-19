import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)

        # 创建最小化配置
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('snapshots', exist_ok=True)
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
  interval_hours: 6
  min_samples: 100
memory:
  max_short_term_samples: 100
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        agent = AgentLoop('config.yaml')
        self.assertEqual(agent._cycle_count, 0)
        self.assertFalse(agent._running)

    def test_run_cycle(self):
        agent = AgentLoop('config.yaml')
        result = agent.run_cycle()

        self.assertEqual(result['cycle'], 1)
        self.assertIn('state', result)
        self.assertIn('predictions', result)
        self.assertIn('errors', result)
        self.assertIn('reward', result)
        self.assertIn('execution', result)
        self.assertIn('safety', result)

        # 验证预测结构
        preds = result['predictions']
        self.assertIn('low', preds)
        self.assertIn('mid', preds)
        self.assertIn('high', preds)

        # 验证执行结果
        self.assertIn(result['execution']['status'], ['executed', 'no_task'])

        agent.stop()

    def test_multiple_cycles(self):
        agent = AgentLoop('config.yaml')
        for _ in range(3):
            result = agent.run_cycle()
            self.assertTrue(result['cycle'] > 0)

        self.assertEqual(agent._cycle_count, 3)
        stats = agent.get_stats()
        self.assertEqual(stats['cycle_count'], 3)
        agent.stop()

    def test_emergency_mode(self):
        agent = AgentLoop('config.yaml')
        # 将资源限制设极低以触发紧急模式
        agent.resource_guardian.max_cpu_percent = 0.001
        agent.resource_guardian.max_mem_percent = 0.001
        result = agent.run_cycle()

        self.assertTrue(result['safety']['emergency_mode'])
        self.assertEqual(result['execution']['status'], 'emergency')
        agent.stop()

    def test_task_pool_integration(self):
        agent = AgentLoop('config.yaml')
        agent.run_cycle()
        stats = agent.task_pool.get_stats()
        self.assertGreaterEqual(stats['total_tasks'], 0)
        agent.stop()

    def test_memory_integration(self):
        agent = AgentLoop('config.yaml')
        agent.run_cycle()
        count = agent.memory.get_sample_count()
        self.assertGreaterEqual(count, 1)
        agent.stop()

    def test_stop(self):
        agent = AgentLoop('config.yaml')
        agent.run_cycle()
        agent.stop()
        self.assertFalse(agent._running)


if __name__ == '__main__':
    unittest.main()
