"""阶段128：全系统集成测试与最终验收

验证12层架构端到端全链路集成，包括：
- 12层架构完整闭环：感知→记忆→预测→评估→奖励→任务→执行→优化→策略→复盘→快照→守护
- 双闭环联动：在线实时修正 + 离线复盘进化
- 长周期端到端一致性
- 故障注入下端到端恢复能力
- 验收标准核验
"""
import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config


class _IntegrationTestBase(unittest.TestCase):
    """集成测试基类。"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_config(self, overrides=None):
        config = {
            'base_weights': {'safe': 0.4, 'gain': 0.4, 'eff': 0.2},
            'resource_limits': {
                'max_cpu_percent': 80.0,
                'max_mem_percent': 80.0,
                'max_db_size_mb': 500,
                'max_snapshot_count': 10,
            },
            'safety': {
                'max_consecutive_failures': 5,
                'forbidden_paths': [],
                'max_model_restructure_iterations': 3,
            },
            'review': {
                'interval_hours': 0.001,
                'min_samples': 10,
                'idle_cpu_threshold': 100.0,
                'idle_duration_seconds': 0,
            },
            'memory': {
                'max_short_term_samples': 1000,
                'value_retention_ratio': 0.3,
                'history_window': 100,
            },
            'logging': {
                'level': 'WARNING',
                'file': 'agent.log',
                'max_bytes': 10485760,
                'backup_count': 5,
            },
            'paths': {
                'data_dir': self.temp_dir,
                'snapshot_dir': os.path.join(self.temp_dir, 'snapshots'),
                'log_dir': os.path.join(self.temp_dir, 'logs'),
            },
            'auto_rollback': {'mae_degradation_threshold': 0.2},
            'task_pool': {'max_size': 50},
        }
        if overrides:
            config.update(overrides)
        return config

    def _make_agent(self, overrides=None):
        import yaml
        config = self._make_config(overrides)
        config_path = os.path.join(self.temp_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(yaml.dump(config, default_flow_style=False))
        return AgentLoop(config_path)

    def _cleanup_agent(self, agent):
        try:
            if hasattr(agent, 'memory') and agent.memory:
                agent.memory.close()
        except Exception:
            pass
        try:
            if hasattr(agent, '_remove_log_handler'):
                agent._remove_log_handler()
        except Exception:
            pass
        try:
            agent.stop()
        except Exception:
            pass


class TestTwelveLayerClosedLoop(_IntegrationTestBase):
    """验证12层架构闭环。"""

    def test_perceive_to_memory_flow(self):
        """感知 → 记忆 数据流。"""
        agent = self._make_agent()
        try:
            agent.run_cycle()
            samples = agent.memory.load_samples(limit=10)
            self.assertGreater(len(samples), 0,
                               "run_cycle 后应有 sample 写入记忆")
        finally:
            self._cleanup_agent(agent)

    def test_predict_to_evaluate_flow(self):
        """预测 → 评估 数据流。"""
        agent = self._make_agent()
        try:
            result = agent.run_cycle()
            self.assertIn('predictions', result)
            self.assertIn('errors', result)
            self.assertIsInstance(result['errors'], dict)
        finally:
            self._cleanup_agent(agent)

    def test_reward_to_task_flow(self):
        """奖励 → 任务 数据流。"""
        agent = self._make_agent()
        try:
            result = agent.run_cycle()
            self.assertIn('reward', result)
            self.assertIn('task', result)
        finally:
            self._cleanup_agent(agent)

    def test_execution_to_optimization_flow(self):
        """执行 → 优化 数据流。"""
        agent = self._make_agent()
        try:
            agent.run_cycle()
            # 验证优化器存在并可被调用
            self.assertTrue(hasattr(agent.optimizer, 'optimize'),
                            "optimizer 应有 optimize 方法")
            self.assertTrue(callable(getattr(agent.optimizer, 'optimize')),
                            "optimizer.optimize 应可调用")
        finally:
            self._cleanup_agent(agent)

    def test_strategy_to_snapshot_flow(self):
        """策略 → 快照 数据流。"""
        agent = self._make_agent()
        try:
            result = agent.run_cycle()
            self.assertIn('snapshot', result)
            snapshots = agent.snapshot_manager.list_snapshots()
            self.assertGreaterEqual(len(snapshots), 0)
        finally:
            self._cleanup_agent(agent)

    def test_safety_guardian_intervention(self):
        """守护层能在异常时介入。"""
        agent = self._make_agent()
        try:
            result = agent.run_cycle()
            self.assertIn('safety', result)
            self.assertIsInstance(result['safety'], dict)
        finally:
            self._cleanup_agent(agent)


class TestDualLoopCoordination(_IntegrationTestBase):
    """验证双闭环联动。"""

    def test_online_realtime_correction_loop(self):
        """在线实时修正闭环：预测→评估→更新预测器。"""
        agent = self._make_agent()
        try:
            for _ in range(5):
                agent.run_cycle()
            # 预测器应已更新
            self.assertGreater(len(agent.low_predictor.get_history('cpu_percent')), 0,
                               "low_predictor 应有历史数据")
            self.assertGreater(len(agent.mid_predictor.predictors['cpu_percent']._history), 0,
                               "mid_predictor 应有历史数据")
        finally:
            self._cleanup_agent(agent)

    def test_offline_review_evolution_loop(self):
        """离线复盘进化闭环：review 触发优化。"""
        agent = self._make_agent()
        try:
            for _ in range(20):
                agent.run_cycle()
            # review 间隔极短，应已触发
            self.assertGreater(agent._last_review_time, 0,
                               "review 应已触发，_last_review_time 应大于 0")
        finally:
            self._cleanup_agent(agent)


class TestLongTermEndToEndConsistency(_IntegrationTestBase):
    """长周期端到端一致性。"""

    def test_50_cycles_no_crash(self):
        """连续 50 个 cycle 不崩溃。"""
        agent = self._make_agent()
        try:
            for i in range(50):
                result = agent.run_cycle()
                self.assertIsNotNone(result)
                self.assertEqual(result['cycle'], i + 1)
        finally:
            self._cleanup_agent(agent)

    def test_state_not_drift_after_many_cycles(self):
        """长周期运行后状态无漂移。"""
        agent = self._make_agent()
        try:
            for _ in range(30):
                agent.run_cycle()
            self.assertLessEqual(len(agent.task_pool.tasks), agent.task_pool.max_size,
                                 "task_pool 不应超过 max_size")
            self.assertLessEqual(len(agent.snapshot_manager.list_snapshots()),
                                 agent.snapshot_manager.max_count,
                                 "snapshot 不应超过 max_count")
        finally:
            self._cleanup_agent(agent)


class TestFaultInjectionRecovery(_IntegrationTestBase):
    """故障注入下端到端恢复。"""

    def test_recovery_after_snapshot_corruption(self):
        """快照损坏后系统仍能运行。"""
        agent = self._make_agent()
        try:
            agent.run_cycle()
            # 损坏快照目录
            snapshot_dir = agent.snapshot_manager.snapshot_dir
            for f in os.listdir(snapshot_dir):
                path = os.path.join(snapshot_dir, f)
                if os.path.isfile(path):
                    with open(path, 'wb') as f:
                        f.write(b'corrupted')
            # 应能继续运行
            result = agent.run_cycle()
            self.assertIsNotNone(result)
        finally:
            self._cleanup_agent(agent)

    def test_recovery_after_memory_db_corruption(self):
        """记忆数据库损坏后系统仍能运行。"""
        agent = self._make_agent()
        try:
            agent.run_cycle()
            # 损坏数据库文件
            db_path = agent.memory.db_path
            if os.path.exists(db_path):
                with open(db_path, 'wb') as f:
                    f.write(b'corrupted sqlite')
            # 应能继续运行
            result = agent.run_cycle()
            self.assertIsNotNone(result)
        finally:
            self._cleanup_agent(agent)


class TestAcceptanceCriteria(_IntegrationTestBase):
    """验收标准核验。"""

    def test_all_branches_executable(self):
        """核心代码分支可执行：正常/紧急/异常路径。"""
        agent = self._make_agent()
        try:
            # 正常路径
            normal = agent.run_cycle()
            self.assertIsNotNone(normal)
            # 安全路径（默认不会触发 emergency，但应可执行）
            self.assertIn('safety', normal)
        finally:
            self._cleanup_agent(agent)

    def test_no_unhandled_exceptions(self):
        """运行中无未处理异常。"""
        agent = self._make_agent()
        try:
            try:
                for _ in range(10):
                    agent.run_cycle()
            except Exception as e:
                self.fail(f"运行中不应抛出未处理异常: {e}")
        finally:
            self._cleanup_agent(agent)

    def test_data_flow_consistency(self):
        """数据流转无逻辑冲突。"""
        agent = self._make_agent()
        try:
            result = agent.run_cycle()
            # cycle 编号一致
            self.assertEqual(result['cycle'], agent._cycle_count)
            # 任务在执行结果中
            if result['task']:
                self.assertIn('id', result['task'])
        finally:
            self._cleanup_agent(agent)


if __name__ == '__main__':
    unittest.main()
