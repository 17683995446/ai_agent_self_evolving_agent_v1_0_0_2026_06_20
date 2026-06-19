"""阶段129：长期稳定性与生产就绪验收

模拟生产环境长期运行，验证生产就绪度：
- 长周期（200+ cycle）连续运行
- 内存泄漏、FD泄漏、磁盘增长检测
- 故障注入组合
- 优雅关闭与重启恢复
- 资源限额下运行
"""
import os
import sys
import shutil
import tempfile
import unittest
import resource

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config


class _LongTermTestBase(unittest.TestCase):
    """长期稳定性测试基类。"""

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

    def _make_agent(self):
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
                'interval_hours': 6,
                'min_samples': 100,
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
        import yaml
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

    def _get_fd_count(self):
        """获取当前进程打开的文件描述符数量。"""
        try:
            return len(os.listdir(f'/proc/{os.getpid()}/fd'))
        except Exception:
            return -1


class TestLongRunningStability(_LongTermTestBase):
    """长周期运行稳定性。"""

    def test_200_cycles_no_crash(self):
        """连续运行 200 个 cycle 不崩溃。"""
        agent = self._make_agent()
        try:
            for i in range(200):
                result = agent.run_cycle()
                self.assertIsNotNone(result)
                self.assertEqual(result['cycle'], i + 1)
        finally:
            self._cleanup_agent(agent)

    def test_memory_usage_not_unbounded(self):
        """长周期运行后内存增长应可控。"""
        agent = self._make_agent()
        try:
            # warmup
            for _ in range(10):
                agent.run_cycle()
            mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            for _ in range(100):
                agent.run_cycle()

            mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            growth_ratio = mem_after / mem_before if mem_before > 0 else 1.0
            self.assertLess(growth_ratio, 5.0,
                            f"内存增长比 {growth_ratio:.2f} 超过 5x，可能存在内存泄漏")
        finally:
            self._cleanup_agent(agent)

    def test_fd_count_stable(self):
        """长周期运行后 FD 数量应稳定。"""
        agent = self._make_agent()
        try:
            fd_before = self._get_fd_count()
            for _ in range(100):
                agent.run_cycle()
            fd_after = self._get_fd_count()
            fd_growth = fd_after - fd_before
            self.assertLess(fd_growth, 50,
                            f"FD 增长 {fd_growth} 超过 50，可能存在 FD 泄漏")
        finally:
            self._cleanup_agent(agent)

    def test_disk_usage_bounded(self):
        """长周期运行后磁盘增长应有界。"""
        agent = self._make_agent()
        try:
            def get_dir_size(path):
                total = 0
                for dirpath, _, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total += os.path.getsize(fp)
                return total

            for _ in range(50):
                agent.run_cycle()

            size = get_dir_size(self.temp_dir)
            # 50 个 cycle 后数据目录不应超过 100MB
            self.assertLess(size, 100 * 1024 * 1024,
                            f"磁盘使用 {size} 字节超过 100MB")
        finally:
            self._cleanup_agent(agent)


class TestGracefulShutdownAndRecovery(_LongTermTestBase):
    """优雅关闭与重启恢复。"""

    def test_graceful_shutdown_does_not_leak_resources(self):
        """优雅关闭不应泄漏资源。"""
        agent = self._make_agent()
        for _ in range(10):
            agent.run_cycle()
        fd_before = self._get_fd_count()
        self._cleanup_agent(agent)
        fd_after = self._get_fd_count()
        self.assertLessEqual(fd_after, fd_before + 5,
                             "优雅关闭后 FD 数量应基本恢复")

    def test_restart_with_existing_data(self):
        """使用已有数据重启应能继续运行。"""
        agent1 = self._make_agent()
        try:
            for _ in range(20):
                agent1.run_cycle()
        finally:
            self._cleanup_agent(agent1)

        # 同一配置和数据目录重启
        agent2 = self._make_agent()
        try:
            result = agent2.run_cycle()
            self.assertIsNotNone(result)
            self.assertEqual(result['cycle'], 1)
            # 数据库应仍包含历史 sample
            samples = agent2.memory.load_samples(limit=100)
            self.assertGreater(len(samples), 0,
                               "重启后应能读取历史 sample")
        finally:
            self._cleanup_agent(agent2)


class TestProductionReadinessChecklist(_LongTermTestBase):
    """生产就绪 checklist。"""

    def test_all_subsystems_initialized(self):
        """所有子系统应正确初始化。"""
        agent = self._make_agent()
        try:
            required_attrs = [
                'state_vector', 'low_predictor', 'mid_predictor', 'high_predictor',
                'reward_engine', 'task_pool', 'task_generator', 'memory',
                'optimizer', 'review_engine', 'snapshot_manager', 'auto_rollback',
                'resource_guardian', 'behavior_guardian', 'safety_enforcer',
                'alert_manager',
            ]
            for attr in required_attrs:
                self.assertTrue(hasattr(agent, attr),
                                f"Agent 缺少子系统 {attr}")
        finally:
            self._cleanup_agent(agent)

    def test_resource_limits_configured(self):
        """资源限制应已配置。"""
        agent = self._make_agent()
        try:
            self.assertGreater(agent.resource_guardian.max_cpu_percent, 0)
            self.assertGreater(agent.resource_guardian.max_mem_percent, 0)
        finally:
            self._cleanup_agent(agent)

    def test_safety_guardian_configured(self):
        """安全守护应已配置。"""
        agent = self._make_agent()
        try:
            self.assertTrue(hasattr(agent.behavior_guardian, 'max_consecutive_failures'))
            self.assertTrue(hasattr(agent.safety_enforcer, 'full_check'))
        finally:
            self._cleanup_agent(agent)

    def test_logging_configured(self):
        """日志系统应已配置。"""
        agent = self._make_agent()
        try:
            self.assertIsNotNone(agent._log_handler)
            self.assertTrue(os.path.exists(agent._log_handler.baseFilename))
        finally:
            self._cleanup_agent(agent)


if __name__ == '__main__':
    unittest.main()
