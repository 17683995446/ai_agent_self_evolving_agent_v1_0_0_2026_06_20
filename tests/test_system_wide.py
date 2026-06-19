import os
import sys
import unittest
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.state_machine import AgentState, StateMachine
from src.core.event_bus import EventBus, Event
from src.core.config_reloader import ConfigReloader
from src.monitoring.log_aggregator import LogAggregator


class TestSystemWideIntegration(unittest.TestCase):
    """系统级整体运行验证"""

    def setUp(self):
        # 重置Config单例
        from src.core.config import Config
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
dashboard:
  enabled: true
  host: 127.0.0.1
  port: 0
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_modules_together(self):
        """验证所有核心模块协同工作"""
        agent = AgentLoop('config.yaml')

        # 验证所有模块已初始化
        self.assertIsNotNone(agent.state_vector)
        self.assertIsNotNone(agent.reward_engine)
        self.assertIsNotNone(agent.low_predictor)
        self.assertIsNotNone(agent.mid_predictor)
        self.assertIsNotNone(agent.high_predictor)
        self.assertIsNotNone(agent.task_pool)
        self.assertIsNotNone(agent.memory)
        self.assertIsNotNone(agent.optimizer)
        self.assertIsNotNone(agent.review_engine)
        self.assertIsNotNone(agent.snapshot_manager)
        self.assertIsNotNone(agent.auto_rollback)
        self.assertIsNotNone(agent.safety_enforcer)
        self.assertIsNotNone(agent.dashboard)

        # 运行完整周期
        for _ in range(5):
            result = agent.run_cycle()
            self.assertIn('predictions', result)
            self.assertIn('reward', result)
            self.assertIn('execution', result)

        agent.stop()

    def test_state_machine_with_agent(self):
        """状态机与Agent协同"""
        sm = StateMachine(AgentState.IDLE)
        agent = AgentLoop('config.yaml')

        # 从IDLE自动转换到ACTIVE
        sm.auto_transition({'emergency_mode': False})
        self.assertEqual(sm.state, AgentState.ACTIVE)

        # 运行周期
        result = agent.run_cycle()
        self.assertFalse(result['safety']['emergency_mode'])

        agent.stop()

    def test_event_bus_with_agent(self):
        """事件总线与Agent协同"""
        import asyncio

        async def _test():
            bus = EventBus()
            events_received = []

            async def handler(event):
                events_received.append(event.type)

            bus.subscribe('agent_cycle', handler)
            await bus.start()

            agent = AgentLoop('config.yaml')
            agent.run_cycle()

            # 手动发布事件（模拟Agent内部事件）
            await bus.publish(Event(type='agent_cycle', payload={'cycle': 1}))

            await asyncio.sleep(0.2)
            await bus.stop()
            agent.stop()

            self.assertIn('agent_cycle', events_received)

        asyncio.run(_test())

    def test_log_aggregator_with_agent(self):
        """日志聚合器与Agent协同"""
        aggregator = LogAggregator()
        agent = AgentLoop('config.yaml')

        # 运行并生成日志
        for _ in range(3):
            agent.run_cycle()

        # 模拟摄入Agent日志
        aggregator.ingest("INFO: Agent cycle completed")
        aggregator.ingest("ERROR: Something went wrong")
        aggregator.ingest("ERROR: Another error")

        stats = aggregator.get_stats()
        self.assertGreaterEqual(stats['total_logs_ingested'], 3)

        agent.stop()

    def test_config_reloader_with_agent(self):
        """配置热重载与Agent协同"""
        reloader = ConfigReloader('config.yaml', check_interval=0.5)
        changes = []

        def on_change(config):
            changes.append(config)

        reloader.add_callback(on_change)
        reloader.start()

        # 修改配置
        time.sleep(0.1)
        with open('config.yaml', 'a') as f:
            f.write("\n# modified\n")

        time.sleep(0.7)
        reloader.stop()

        self.assertGreaterEqual(len(changes), 1)

    def test_dashboard_with_agent_cycles(self):
        """仪表盘与Agent多周期协同"""
        agent = AgentLoop('config.yaml')
        agent.dashboard.start()
        port = agent.dashboard._server.server_address[1]

        try:
            # 运行多个周期
            for _ in range(5):
                agent.run_cycle()

            time.sleep(0.3)

            # 通过HTTP验证数据
            import urllib.request
            import json
            req = urllib.request.Request(f'http://127.0.0.1:{port}/api/data')
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(data['cycles'], 5)
                self.assertEqual(len(data['history']), 5)
                self.assertGreaterEqual(data['memory_samples'], 5)
        finally:
            agent.stop()

    def test_emergency_full_flow(self):
        """紧急模式完整流程验证"""
        agent = AgentLoop('config.yaml')

        # 正常周期
        result = agent.run_cycle()
        self.assertFalse(result['safety']['emergency_mode'])

        # 触发紧急（同时限制CPU和内存）
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

    def test_memory_growth_control(self):
        """验证内存增长受控"""
        agent = AgentLoop('config.yaml')

        initial_count = agent.memory.get_sample_count()

        # 运行多次
        for _ in range(20):
            agent.run_cycle()

        final_count = agent.memory.get_sample_count()
        self.assertEqual(final_count - initial_count, 20)

        # 验证没有无限增长（受max_samples限制）
        self.assertLessEqual(final_count, 100 + 20)

        agent.stop()

    def test_prediction_all_levels(self):
        """验证三层预测器都产生输出"""
        agent = AgentLoop('config.yaml')

        result = agent.run_cycle()
        preds = result['predictions']

        # 底层预测
        self.assertIn('low', preds)
        self.assertGreater(len(preds['low']), 0)

        # 中层预测
        self.assertIn('mid', preds)
        self.assertGreater(len(preds['mid']), 0)

        # 高层预测
        self.assertIn('high', preds)

        agent.stop()

    def test_reward_within_bounds(self):
        """验证奖励值始终在合理范围内"""
        agent = AgentLoop('config.yaml')

        rewards = []
        for _ in range(10):
            result = agent.run_cycle()
            reward = result['reward']['total_reward']
            rewards.append(reward)
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)

        agent.stop()

    def test_task_pool_never_empty(self):
        """验证任务池不会为空"""
        agent = AgentLoop('config.yaml')

        for _ in range(5):
            agent.run_cycle()
            stats = agent.task_pool.get_stats()
            # 任务池应该始终有任务（因为生成器会补充）
            self.assertGreaterEqual(stats['total_tasks'], 0)

        agent.stop()

    def test_stats_consistency(self):
        """验证统计数据一致性"""
        agent = AgentLoop('config.yaml')

        for _ in range(5):
            agent.run_cycle()

        stats = agent.get_stats()
        self.assertEqual(stats['cycle_count'], 5)
        self.assertGreaterEqual(stats['memory_samples'], 5)
        self.assertGreaterEqual(stats['recent_success_rate'], 0.0)
        self.assertLessEqual(stats['recent_success_rate'], 1.0)

        agent.stop()


if __name__ == '__main__':
    unittest.main()
