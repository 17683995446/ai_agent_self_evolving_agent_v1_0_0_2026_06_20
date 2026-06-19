"""阶段125：配置热加载与运行时变更安全测试

验证配置系统的正确性、热加载能力、参数校验和变更传播。
暴露并验证以下bug的修复：
1. Config.reload() 方法缺失（test_config_boundary.py 中的 tautology 测试掩盖了此问题）
2. state_vector.history_window 键路径不匹配（配置值永远不生效）
3. TaskPool.max_size / AutoRollback.mae_degradation_threshold 硬编码
4. 配置参数无校验（负数/错误类型/超范围值静默接受）
"""
import os
import sys
import math
import time
import shutil
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config
from src.core.config_reloader import ConfigReloader
from src.perception.state_vector import StateVector


def _write_config(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _make_config_content(**overrides):
    """生成完整配置内容，支持覆盖指定字段。"""
    config = {
        'base_weights': {'safe': 0.4, 'gain': 0.4, 'eff': 0.2},
        'resource_limits': {
            'max_cpu_percent': 20.0,
            'max_mem_percent': 20.0,
            'max_db_size_mb': 500,
            'max_snapshot_count': 10,
        },
        'safety': {
            'max_consecutive_failures': 5,
            'forbidden_paths': ['/boot', '/etc/shadow'],
            'max_model_restructure_iterations': 3,
        },
        'review': {
            'interval_hours': 6,
            'min_samples': 100,
            'idle_cpu_threshold': 10.0,
            'idle_duration_seconds': 300,
        },
        'memory': {
            'max_short_term_samples': 10000,
            'value_retention_ratio': 0.3,
            'history_window': 100,
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/agent.log',
            'max_bytes': 10485760,
            'backup_count': 5,
        },
        'paths': {
            'data_dir': 'data',
            'snapshot_dir': 'snapshots',
            'log_dir': 'logs',
        },
    }
    config.update(overrides)
    import yaml
    return yaml.dump(config, default_flow_style=False)


class _ConfigTestBase(unittest.TestCase):
    """配置测试基类，管理临时目录和单例重置。"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.yaml')

    def tearDown(self):
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ============================================================
# Bug 1: Config.reload() 方法缺失
# ============================================================

class TestConfigReloadMethod(_ConfigTestBase):
    """验证 Config.reload() 方法的存在与正确性。

    之前 test_config_boundary.py 中调用 cfg.reload() 但该方法不存在，
    被 try/except Exception: pass 静默吞掉，测试变成 tautology。
    """

    def test_reload_method_exists(self):
        """Config 应该有 reload 方法。"""
        _write_config(self.config_path, 'key1: value1\n')
        cfg = Config(config_path=self.config_path)
        self.assertTrue(hasattr(cfg, 'reload'),
                        "Config 缺少 reload 方法，test_config_boundary.py 中的测试是 tautology")

    def test_reload_picks_up_new_values(self):
        """reload 后应读取到文件中的新值。"""
        _write_config(self.config_path, 'key1: value1\n')
        cfg = Config(config_path=self.config_path)
        self.assertEqual(cfg.get('key1'), 'value1')

        time.sleep(0.05)
        _write_config(self.config_path, 'key1: value2\nkey2: 42\n')
        cfg.reload()

        self.assertEqual(cfg.get('key1'), 'value2')
        self.assertEqual(cfg.get('key2'), 42)

    def test_reload_preserves_old_config_on_file_deletion(self):
        """文件删除后 reload 应保留旧配置，不崩溃。"""
        _write_config(self.config_path, 'key1: value1\n')
        cfg = Config(config_path=self.config_path)
        self.assertEqual(cfg.get('key1'), 'value1')

        os.remove(self.config_path)
        # reload 不应崩溃，且应保留旧值
        try:
            cfg.reload()
        except FileNotFoundError:
            # 可接受：抛出 FileNotFoundError，但旧配置应保留
            pass
        self.assertEqual(cfg.get('key1'), 'value1',
                         "文件删除后 reload 不应清空已有配置")

    def test_reload_preserves_old_config_on_corruption(self):
        """文件损坏后 reload 应保留旧配置，不崩溃。"""
        _write_config(self.config_path, 'key1: value1\n')
        cfg = Config(config_path=self.config_path)
        self.assertEqual(cfg.get('key1'), 'value1')

        _write_config(self.config_path, '{{bad yaml')
        # reload 不应崩溃，且应保留旧值
        try:
            cfg.reload()
        except Exception:
            pass
        self.assertEqual(cfg.get('key1'), 'value1',
                         "文件损坏后 reload 不应清空已有配置")

    def test_reload_is_thread_safe(self):
        """并发 reload 和 get 应线程安全。"""
        _write_config(self.config_path, 'counter: 0\n')
        cfg = Config(config_path=self.config_path)
        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    cfg.get('counter')
                    cfg.to_dict()
                except Exception as e:
                    errors.append(str(e))

        def reloader():
            i = 0
            while not stop.is_set():
                try:
                    _write_config(self.config_path, f'counter: {i}\n')
                    cfg.reload()
                    i += 1
                except Exception as e:
                    errors.append(str(e))
                time.sleep(0.001)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=reloader)
        t1.start()
        t2.start()
        time.sleep(0.5)
        stop.set()
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertEqual(len(errors), 0, f"并发 reload/get 出错: {errors[:3]}")


# ============================================================
# Bug 2: state_vector.history_window 键路径不匹配
# ============================================================

class TestStateVectorConfigPropagation(_ConfigTestBase):
    """验证 StateVector 正确读取 memory.history_window 配置。

    之前 state_vector.py 读取 self.config.get('history_window', 100)，
    但配置中 history_window 位于 memory.history_window，
    导致配置值永远不生效。
    """

    def test_history_window_from_memory_section(self):
        """StateVector 应从 memory.history_window 读取配置值。"""
        config = {
            'memory': {'history_window': 50, 'max_short_term_samples': 10000},
        }
        import yaml
        _write_config(self.config_path, yaml.dump(config))
        cfg = Config(config_path=self.config_path)
        cfg_dict = cfg.to_dict()

        sv = StateVector(cfg_dict)
        # 应读取到 50，而非默认 100
        self.assertEqual(sv._history_window, 50,
                         "StateVector 未从 memory.history_window 读取配置，使用了默认值")

    def test_history_window_fallback_to_default(self):
        """配置中无 memory.history_window 时应回退到默认值 100。"""
        config = {'other_key': 'value'}
        import yaml
        _write_config(self.config_path, yaml.dump(config))
        cfg = Config(config_path=self.config_path)

        sv = StateVector(cfg.to_dict())
        self.assertEqual(sv._history_window, 100)

    def test_history_window_affects_mae_history_size(self):
        """history_window 应实际限制 mae_history 的长度。"""
        config = {'memory': {'history_window': 5}}
        import yaml
        _write_config(self.config_path, yaml.dump(config))
        cfg = Config(config_path=self.config_path)

        sv = StateVector(cfg.to_dict())
        for i in range(20):
            sv.update_mae(float(i))

        self.assertEqual(len(sv._recent_mae_history), 5,
                         "history_window=5 应限制 mae_history 长度为 5")

    def test_history_window_affects_task_success_size(self):
        """history_window 应实际限制 task_success_history 的长度。"""
        config = {'memory': {'history_window': 3}}
        import yaml
        _write_config(self.config_path, yaml.dump(config))
        cfg = Config(config_path=self.config_path)

        sv = StateVector(cfg.to_dict())
        for i in range(10):
            sv.update_task_success(i % 2 == 0)

        self.assertEqual(len(sv._recent_task_success), 3,
                         "history_window=3 应限制 task_success_history 长度为 3")


# ============================================================
# Bug 3: 硬编码参数未暴露为配置
# ============================================================

class TestHardcodedParamsExposed(_ConfigTestBase):
    """验证 TaskPool.max_size 和 AutoRollback.mae_degradation_threshold
    可以从配置读取，而非硬编码。"""

    def test_task_pool_max_size_configurable(self):
        """TaskPool 的 max_size 应可从配置读取。"""
        from src.task.task_pool import TaskPool
        # 直接验证 TaskPool 接受 max_size 参数
        pool = TaskPool(max_size=10)
        self.assertEqual(pool.max_size, 10)

        pool2 = TaskPool(max_size=200)
        self.assertEqual(pool2.max_size, 200)

    def test_auto_rollback_threshold_configurable(self):
        """AutoRollback 的 mae_degradation_threshold 应可从配置读取。"""
        from src.optimization.auto_rollback import AutoRollback
        from src.optimization.snapshot_manager import SnapshotManager

        temp_dir = tempfile.mkdtemp()
        try:
            sm = SnapshotManager(snapshot_dir=temp_dir)
            ar = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=0.5)
            self.assertEqual(ar.mae_degradation_threshold, 0.5)

            ar2 = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=0.1)
            self.assertEqual(ar2.mae_degradation_threshold, 0.1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_agent_loop_reads_task_pool_max_size_from_config(self):
        """AgentLoop 应从配置读取 task_pool.max_size。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(task_pool={'max_size': 5})
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            self.assertEqual(agent.task_pool.max_size, 5,
                             "AgentLoop 未从配置读取 task_pool.max_size，使用了硬编码 50")
        finally:
            agent.stop()
            self._cleanup_agent(agent)

    def test_agent_loop_reads_auto_rollback_threshold_from_config(self):
        """AgentLoop 应从配置读取 auto_rollback.mae_degradation_threshold。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(auto_rollback={'mae_degradation_threshold': 0.5})
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            self.assertAlmostEqual(
                agent.auto_rollback.mae_degradation_threshold, 0.5,
                msg="AgentLoop 未从配置读取 auto_rollback.mae_degradation_threshold，使用了硬编码 0.2"
            )
        finally:
            agent.stop()
            self._cleanup_agent(agent)

    def _cleanup_agent(self, agent):
        """清理 agent 持有的资源。"""
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


# ============================================================
# Bug 4: 配置参数无校验
# ============================================================

class TestConfigValidation(_ConfigTestBase):
    """验证配置参数的校验逻辑。

    之前 Config.load() 无任何校验，负数/错误类型/超范围值静默接受。
    """

    def test_negative_max_samples_sanitized(self):
        """memory.max_short_term_samples 为负数时应被校正。"""
        from src.memory.short_term import ShortTermMemory

        config = _make_config_content(memory={
            'max_short_term_samples': -100,
            'history_window': 100,
        })
        _write_config(self.config_path, config)

        db_path = os.path.join(self.temp_dir, 'test.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=-100)
        try:
            # 负数应被校正为正数（至少 1）
            self.assertGreater(mem.max_samples, 0,
                               "max_samples 为负数时应被校正为正数")
        finally:
            mem.close()

    def test_zero_max_samples_sanitized(self):
        """memory.max_short_term_samples 为 0 时应被校正。"""
        from src.memory.short_term import ShortTermMemory

        db_path = os.path.join(self.temp_dir, 'test.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=0)
        try:
            self.assertGreater(mem.max_samples, 0,
                               "max_samples 为 0 时应被校正为正数")
        finally:
            mem.close()

    def test_negative_task_pool_max_size_sanitized(self):
        """TaskPool.max_size 为负数时应被校正。"""
        from src.task.task_pool import TaskPool

        pool = TaskPool(max_size=-10)
        self.assertGreater(pool.max_size, 0,
                           "max_size 为负数时应被校正为正数")

    def test_zero_task_pool_max_size_sanitized(self):
        """TaskPool.max_size 为 0 时应被校正。"""
        from src.task.task_pool import TaskPool

        pool = TaskPool(max_size=0)
        self.assertGreater(pool.max_size, 0,
                           "max_size 为 0 时应被校正为正数")

    def test_negative_degradation_threshold_sanitized(self):
        """AutoRollback.mae_degradation_threshold 为负数时应被校正。"""
        from src.optimization.auto_rollback import AutoRollback
        from src.optimization.snapshot_manager import SnapshotManager

        temp_dir = tempfile.mkdtemp()
        try:
            sm = SnapshotManager(snapshot_dir=temp_dir)
            ar = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=-0.5)
            self.assertGreaterEqual(ar.mae_degradation_threshold, 0,
                                    "mae_degradation_threshold 为负数时应被校正为非负数")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_negative_review_interval_sanitized(self):
        """review.interval_hours 为负数时 AgentLoop 应校正。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(review={'interval_hours': -1, 'min_samples': 1})
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            self.assertGreaterEqual(agent._review_interval, 0,
                                    "review.interval_hours 为负数时应被校正为非负数")
        finally:
            agent.stop()
            self._cleanup_agent(agent)

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


# ============================================================
# 配置热加载与运行时变更
# ============================================================

class TestConfigHotReload(_ConfigTestBase):
    """验证 ConfigReloader 的热加载机制。"""

    def test_force_reload_returns_new_config(self):
        """force_reload 应返回新的配置字典。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)

        time.sleep(0.05)
        _write_config(self.config_path, 'key1: value2\nkey2: 42\n')
        new_config = reloader.force_reload()

        self.assertIsNotNone(new_config)
        self.assertEqual(new_config.get('key1'), 'value2')
        self.assertEqual(new_config.get('key2'), 42)

    def test_callback_receives_new_config(self):
        """回调应接收到新的配置字典。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        received = []

        reloader.add_callback(lambda cfg: received.append(cfg))
        reloader.start()

        try:
            time.sleep(0.05)
            _write_config(self.config_path, 'key1: value2\n')
            time.sleep(0.5)  # 等待 watch_loop 检测

            self.assertGreater(len(received), 0, "回调未被触发")
            self.assertEqual(received[-1].get('key1'), 'value2')
        finally:
            reloader.stop()

    def test_callback_exception_isolation(self):
        """单个回调异常不应影响其他回调。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        results = []

        def bad_callback(cfg):
            raise RuntimeError("intentional error")

        def good_callback(cfg):
            results.append(cfg)

        reloader.add_callback(bad_callback)
        reloader.add_callback(good_callback)

        # force_reload 应不因 bad_callback 异常而中断
        new_config = reloader.force_reload()
        self.assertIsNotNone(new_config)
        self.assertEqual(len(results), 1, "好回调应被正常执行")

    def test_reloader_handles_file_deletion(self):
        """文件删除时 reloader 不应崩溃。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        reloader.start()

        try:
            os.remove(self.config_path)
            time.sleep(0.3)  # 等待 watch_loop 检测
            # reloader 应仍在运行
            self.assertTrue(reloader.is_running())
        finally:
            reloader.stop()

    def test_reloader_handles_corrupted_file(self):
        """文件损坏时 reloader 不应崩溃，回调不触发。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        received = []
        reloader.add_callback(lambda cfg: received.append(cfg))
        reloader.start()

        try:
            time.sleep(0.05)
            _write_config(self.config_path, '{{bad yaml')
            time.sleep(0.3)
            # 损坏文件不应触发回调（_load_config_file 返回 None）
            self.assertEqual(len(received), 0, "损坏文件不应触发回调")
            self.assertTrue(reloader.is_running())
        finally:
            reloader.stop()

    def test_reloader_restores_after_file_recreated(self):
        """文件删除后重建，reloader 应恢复检测。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        received = []
        reloader.add_callback(lambda cfg: received.append(cfg))
        reloader.start()

        try:
            os.remove(self.config_path)
            time.sleep(0.2)

            _write_config(self.config_path, 'key1: value2\n')
            time.sleep(0.5)

            self.assertGreater(len(received), 0, "文件重建后回调应被触发")
            self.assertEqual(received[-1].get('key1'), 'value2')
        finally:
            reloader.stop()

    def test_check_interval_zero_does_not_crash(self):
        """check_interval=0 不应导致死循环或崩溃。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.0)
        reloader.start()
        try:
            time.sleep(0.1)
            self.assertTrue(reloader.is_running())
        finally:
            reloader.stop()

    def test_remove_callback(self):
        """移除回调后不再被通知。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        received = []

        cb = lambda cfg: received.append(cfg)
        reloader.add_callback(cb)
        self.assertTrue(reloader.remove_callback(cb))

        reloader.force_reload()
        self.assertEqual(len(received), 0, "移除的回调不应被触发")

    def test_remove_nonexistent_callback(self):
        """移除不存在的回调应返回 False。"""
        _write_config(self.config_path, 'key1: value1\n')
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        cb = lambda cfg: None
        self.assertFalse(reloader.remove_callback(cb))


# ============================================================
# 配置变更与运行中状态的交互
# ============================================================

class TestConfigChangeDuringRuntime(_ConfigTestBase):
    """验证配置变更对运行中组件的影响。"""

    def test_config_snapshot_isolation(self):
        """AgentLoop 的 cfg 快照应与 Config 单例隔离。"""
        _write_config(self.config_path, 'key1: value1\n')
        cfg = Config(config_path=self.config_path)
        snapshot = cfg.to_dict()

        # 修改单例
        cfg.set('key1', 'changed')
        self.assertEqual(cfg.get('key1'), 'changed')

        # 快照应不受影响
        self.assertEqual(snapshot.get('key1'), 'value1',
                         "to_dict() 返回的快照应与后续修改隔离")

    def test_concurrent_load_and_get(self):
        """并发 load（替换整个 _config）与 get 应线程安全。"""
        _write_config(self.config_path, 'counter: 0\n')
        cfg = Config(config_path=self.config_path)
        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    val = cfg.get('counter')
                    if val is not None and not isinstance(val, (int, float)):
                        errors.append(f"unexpected type: {type(val)}")
                    cfg.to_dict()
                except Exception as e:
                    errors.append(str(e))

        def loader():
            i = 0
            while not stop.is_set():
                try:
                    _write_config(self.config_path, f'counter: {i}\n')
                    cfg.load(self.config_path)
                    i += 1
                except Exception as e:
                    errors.append(str(e))
                time.sleep(0.002)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=loader)
        t1.start()
        t2.start()
        time.sleep(0.5)
        stop.set()
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertEqual(len(errors), 0, f"并发 load/get 出错: {errors[:3]}")

    def test_config_set_creates_nested_path(self):
        """set 应正确创建嵌套路径。"""
        _write_config(self.config_path, 'existing: true\n')
        cfg = Config(config_path=self.config_path)

        cfg.set('new.nested.deep', 42)
        self.assertEqual(cfg.get('new.nested.deep'), 42)
        self.assertEqual(cfg.get('new.nested'), {'deep': 42})

    def test_config_set_overwrites_non_dict_intermediate(self):
        """set 在中间节点为非 dict 时应正确处理。"""
        _write_config(self.config_path, 'a: string_value\n')
        cfg = Config(config_path=self.config_path)

        # a 是字符串，set a.b 应将 a 替换为 dict
        cfg.set('a.b', 42)
        self.assertEqual(cfg.get('a.b'), 42)
        self.assertEqual(cfg.get('a'), {'b': 42})


# ============================================================
# 多环境配置一致性
# ============================================================

class TestMultiEnvConfigConsistency(unittest.TestCase):
    """验证 dev/test/prod 三份配置的键名一致性。"""

    def setUp(self):
        self.config_dir = os.path.join(
            os.path.dirname(__file__), '..', 'config'
        )

    def _load_yaml(self, name):
        path = os.path.join(self.config_dir, name)
        if not os.path.exists(path):
            self.skipTest(f"{name} 不存在")
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def test_all_envs_have_same_top_level_keys(self):
        """dev/test/prod 配置应有相同的顶层键集合。"""
        envs = {}
        for name in ['config.yaml', 'config.dev.yaml', 'config.test.yaml', 'config.prod.yaml']:
            try:
                envs[name] = self._load_yaml(name)
            except unittest.SkipTest:
                continue

        if len(envs) < 2:
            self.skipTest("配置文件不足")

        key_sets = {name: set(cfg.keys()) for name, cfg in envs.items()}
        # 取交集
        common = set.intersection(*key_sets.values())
        # 每个环境的额外键
        for name, keys in key_sets.items():
            extra = keys - common
            # 允许环境特有的键，但核心键必须存在
            for required in ['base_weights', 'resource_limits', 'safety', 'memory', 'paths']:
                self.assertIn(required, keys, f"{name} 缺少核心键 {required}")

    def test_base_weights_values_valid(self):
        """所有环境的 base_weights 值应合法（0-1之间）。"""
        for name in ['config.yaml', 'config.dev.yaml', 'config.test.yaml', 'config.prod.yaml']:
            try:
                cfg = self._load_yaml(name)
            except unittest.SkipTest:
                continue
            bw = cfg.get('base_weights', {})
            for key in ['safe', 'gain', 'eff']:
                if key in bw:
                    val = bw[key]
                    self.assertIsInstance(val, (int, float),
                                          f"{name}.base_weights.{key} 类型错误: {type(val)}")
                    self.assertGreaterEqual(val, 0,
                                            f"{name}.base_weights.{key} 不应为负数: {val}")
                    self.assertLessEqual(val, 1,
                                         f"{name}.base_weights.{key} 不应超过 1: {val}")

    def test_resource_limits_values_valid(self):
        """所有环境的 resource_limits 值应合法。"""
        for name in ['config.yaml', 'config.dev.yaml', 'config.test.yaml', 'config.prod.yaml']:
            try:
                cfg = self._load_yaml(name)
            except unittest.SkipTest:
                continue
            rl = cfg.get('resource_limits', {})
            for key in ['max_cpu_percent', 'max_mem_percent']:
                if key in rl:
                    val = rl[key]
                    self.assertIsInstance(val, (int, float),
                                          f"{name}.resource_limits.{key} 类型错误")
                    self.assertGreater(val, 0,
                                       f"{name}.resource_limits.{key} 应大于 0")
                    self.assertLessEqual(val, 100,
                                         f"{name}.resource_limits.{key} 不应超过 100")


# ============================================================
# 端到端配置传播
# ============================================================

class TestConfigPropagationE2E(_ConfigTestBase):
    """验证配置从文件到子系统构造的端到端传播。"""

    def test_custom_base_weights_propagate_to_reward_engine(self):
        """自定义 base_weights 应传播到 RewardEngine。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(base_weights={'safe': 0.6, 'gain': 0.3, 'eff': 0.1})
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            weights = agent.reward_engine.base_weights
            self.assertAlmostEqual(weights[0], 0.6)
            self.assertAlmostEqual(weights[1], 0.3)
            self.assertAlmostEqual(weights[2], 0.1)
        finally:
            agent.stop()
            self._cleanup_agent(agent)

    def test_custom_resource_limits_propagate_to_guardian(self):
        """自定义 resource_limits 应传播到 ResourceGuardian。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(resource_limits={
            'max_cpu_percent': 50.0,
            'max_mem_percent': 60.0,
            'max_db_size_mb': 500,
            'max_snapshot_count': 10,
        })
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            self.assertAlmostEqual(agent.resource_guardian.max_cpu_percent, 50.0)
            self.assertAlmostEqual(agent.resource_guardian.max_mem_percent, 60.0)
        finally:
            agent.stop()
            self._cleanup_agent(agent)

    def test_custom_memory_config_propagates(self):
        """自定义 memory.max_short_term_samples 应传播到 ShortTermMemory。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(
            memory={'max_short_term_samples': 500, 'history_window': 50},
            paths={'data_dir': self.temp_dir, 'snapshot_dir': self.temp_dir, 'log_dir': self.temp_dir},
        )
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            self.assertEqual(agent.memory.max_samples, 500)
            # history_window 应通过 StateVector 传播
            self.assertEqual(agent.state_vector._history_window, 50)
        finally:
            agent.stop()
            self._cleanup_agent(agent)

    def test_custom_safety_config_propagates(self):
        """自定义 safety 配置应传播到 BehaviorGuardian。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(safety={
            'max_consecutive_failures': 10,
            'forbidden_paths': ['/dangerous'],
            'max_model_restructure_iterations': 3,
        })
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            self.assertEqual(agent.behavior_guardian.max_consecutive_failures, 10)
        finally:
            agent.stop()
            self._cleanup_agent(agent)

    def test_custom_review_config_propagates(self):
        """自定义 review 配置应传播到 ReviewEngine。"""
        from src.core.agent_loop import AgentLoop

        config = _make_config_content(review={
            'interval_hours': 2,
            'min_samples': 50,
            'idle_cpu_threshold': 5.0,
            'idle_duration_seconds': 100,
        })
        _write_config(self.config_path, config)

        agent = AgentLoop(self.config_path)
        try:
            self.assertEqual(agent.review_engine.min_samples, 50)
            self.assertEqual(agent._review_interval, 2 * 3600)
        finally:
            agent.stop()
            self._cleanup_agent(agent)

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


if __name__ == '__main__':
    unittest.main()
