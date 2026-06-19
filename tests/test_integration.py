import os
import sys
import unittest
import tempfile
import shutil
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.state_machine import AgentState, StateMachine
from src.core.event_bus import EventBus, Event


class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)

        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('snapshots', exist_ok=True)

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
  max_db_size_mb: 500
  max_snapshot_count: 10
exploration:
  base_ratio: 0.3
  stable_threshold: 0.05
  emergency_ratio: 0.1
  novel_ratio: 0.4
models:
  smoothing_alpha: 0.3
  smoothing_alpha_min: 0.05
  smoothing_alpha_max: 0.8
  arima_order: [1, 1, 1]
  prediction_horizon: 1
safety:
  max_consecutive_failures: 5
  forbidden_paths:
    - /boot
    - /etc
    - /proc
    - /sys
    - /dev
    - /bin
    - /sbin
    - /lib
    - /lib64
    - /usr/bin
    - /usr/sbin
  max_model_restructure_iterations: 3
review:
  interval_hours: 6
  min_samples: 100
  idle_cpu_threshold: 10.0
  idle_duration_seconds: 300
memory:
  max_short_term_samples: 100
  value_retention_ratio: 0.3
  history_window: 100
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log
  max_bytes: 10485760
  backup_count: 5
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_agent_lifecycle(self):
        """测试Agent完整生命周期"""
        agent = AgentLoop('config.yaml')

        # 初始状态
        self.assertEqual(agent._cycle_count, 0)
        stats = agent.get_stats()
        self.assertEqual(stats['cycle_count'], 0)

        # 运行多个周期
        for i in range(5):
            result = agent.run_cycle()
            self.assertEqual(result['cycle'], i + 1)
            self.assertIn('state', result)
            self.assertIn('predictions', result)
            self.assertIn('errors', result)
            self.assertIn('reward', result)
            self.assertIn('execution', result)

        # 验证状态积累
        stats = agent.get_stats()
        self.assertEqual(stats['cycle_count'], 5)
        self.assertGreaterEqual(stats['memory_samples'], 5)

        agent.stop()

    def test_emergency_recovery_flow(self):
        """测试紧急模式触发与恢复流程"""
        agent = AgentLoop('config.yaml')

        # 正常周期
        result = agent.run_cycle()
        self.assertFalse(result['safety']['emergency_mode'])

        # 触发紧急模式
        agent.resource_guardian.max_cpu_percent = 0.001
        agent.resource_guardian.max_mem_percent = 0.001
        result = agent.run_cycle()
        self.assertTrue(result['safety']['emergency_mode'])
        self.assertEqual(result['execution']['status'], 'emergency')

        # 恢复
        agent.resource_guardian.max_cpu_percent = 100.0
        agent.resource_guardian.max_mem_percent = 100.0
        result = agent.run_cycle()
        self.assertFalse(result['safety']['emergency_mode'])

        agent.stop()

    def test_task_pool_evolution(self):
        """测试任务池随时间演化"""
        agent = AgentLoop('config.yaml')

        initial_stats = agent.task_pool.get_stats()

        # 运行多个周期让任务生成
        for _ in range(10):
            agent.run_cycle()

        final_stats = agent.task_pool.get_stats()
        self.assertGreaterEqual(final_stats['total_tasks'], 0)
        self.assertGreaterEqual(final_stats['total_executions'], 0)

        agent.stop()

    def test_memory_persistence(self):
        """测试记忆持久化"""
        agent1 = AgentLoop('config.yaml')
        for _ in range(3):
            agent1.run_cycle()

        sample_count = agent1.memory.get_sample_count()
        self.assertGreaterEqual(sample_count, 3)
        agent1.stop()

        # 重新创建Agent，使用同一个数据库
        agent2 = AgentLoop('config.yaml')
        self.assertEqual(agent2.memory.get_sample_count(), sample_count)
        agent2.stop()

    def test_prediction_improvement(self):
        """测试预测器在多次循环后产生输出"""
        agent = AgentLoop('config.yaml')

        predictions_history = []
        for _ in range(5):
            result = agent.run_cycle()
            predictions_history.append(result['predictions']['low'])

        # 预测器应该始终返回数值
        for preds in predictions_history:
            self.assertIsInstance(preds, dict)
            for metric, value in preds.items():
                self.assertIsInstance(value, float)

        agent.stop()

    def test_reward_computation(self):
        """测试奖励计算在完整链路中工作"""
        agent = AgentLoop('config.yaml')

        result = agent.run_cycle()
        reward = result['reward']

        self.assertIn('total_reward', reward)
        self.assertIn('safe_reward', reward)
        self.assertIn('gain_reward', reward)
        self.assertIn('eff_reward', reward)
        self.assertIn('weights', reward)

        # 奖励应在合理范围内
        self.assertGreaterEqual(reward['total_reward'], 0.0)
        self.assertLessEqual(reward['total_reward'], 1.0)

        agent.stop()

    def test_snapshot_creation(self):
        """测试模型快照在100周期时创建"""
        agent = AgentLoop('config.yaml')

        # 运行99个周期
        for _ in range(99):
            agent.run_cycle()

        # 第100个周期应创建快照
        result = agent.run_cycle()
        self.assertIsNotNone(result['snapshot'])
        self.assertIn('path', result['snapshot'])

        agent.stop()

    def test_consecutive_cycles_consistency(self):
        """测试连续周期的数据一致性"""
        agent = AgentLoop('config.yaml')

        prev_cycle = 0
        for _ in range(10):
            result = agent.run_cycle()
            self.assertEqual(result['cycle'], prev_cycle + 1)
            prev_cycle = result['cycle']

            # 每个周期都应该有状态数据
            self.assertIn('system', result['state'])
            self.assertIn('logs', result['state'])
            self.assertIn('self', result['state'])

        agent.stop()


class TestStateMachineIntegration(unittest.TestCase):
    """状态机集成测试"""

    def test_auto_transition_with_safety(self):
        sm = StateMachine(AgentState.ACTIVE)

        # 模拟安全违规 -> 紧急模式
        transition = sm.auto_transition({'emergency_mode': True})
        self.assertIsNotNone(transition)
        self.assertEqual(sm.state, AgentState.EMERGENCY)

        # 安全恢复 -> 激活模式
        transition = sm.auto_transition({'emergency_mode': False})
        self.assertIsNotNone(transition)
        self.assertEqual(sm.state, AgentState.ACTIVE)

    def test_state_handlers_integration(self):
        sm = StateMachine()
        events = []

        def on_active(from_state, to_state):
            events.append('active')

        def on_emergency(from_state, to_state):
            events.append('emergency')

        sm.on_enter(AgentState.ACTIVE, on_active)
        sm.on_enter(AgentState.EMERGENCY, on_emergency)

        sm.transition(AgentState.ACTIVE, "test")
        sm.transition(AgentState.EMERGENCY, "test")

        self.assertEqual(events, ['active', 'emergency'])


class TestEventBusIntegration(unittest.IsolatedAsyncioTestCase):
    """事件总线集成测试"""

    async def test_event_flow(self):
        bus = EventBus()
        received_events = []

        async def handler(event):
            received_events.append(event.type)

        bus.subscribe('cycle_complete', handler)
        await bus.start()

        await bus.publish(Event(type='cycle_complete', payload={'cycle': 1}))
        await bus.publish(Event(type='cycle_complete', payload={'cycle': 2}))

        await asyncio.sleep(0.2)  # 等待异步处理
        await bus.stop()

        self.assertEqual(len(received_events), 2)


if __name__ == '__main__':
    unittest.main()
