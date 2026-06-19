#!/usr/bin/env python3
"""
配置系统完整性与热重载可靠性测试
目标：验证Config单例安全、热重载稳定、默认值回退正确
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config
from src.core.config_reloader import ConfigReloader


class TestConfigSingletonIntegrity(unittest.TestCase):
    """Config单例完整性测试"""

    def tearDown(self):
        Config._instance = None
        Config._config = {}
        Config._lock = threading.Lock()

    def test_singleton_identity(self):
        """多次创建应返回同一实例"""
        Config._instance = None
        c1 = Config()
        c2 = Config()
        self.assertIs(c1, c2)

    def test_concurrent_creation(self):
        """多线程并发创建不应产生多个实例"""
        Config._instance = None
        instances = []
        lock = threading.Lock()

        def creator():
            c = Config()
            with lock:
                instances.append(c)

        threads = [threading.Thread(target=creator) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(set(id(i) for i in instances)), 1)

    def test_concurrent_get_set(self):
        """多线程并发get/set不应崩溃"""
        Config._instance = None
        Config._config = {'base': {'counter': 0}}
        c = Config()
        errors = []

        def worker():
            try:
                for i in range(100):
                    c.set(f'thread.{i}', i)
                    c.get(f'thread.{i}')
                    c.get('base.counter', 0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_set_overwrite_non_dict(self):
        """set()应能覆盖非dict的中间值"""
        Config._instance = None
        Config._config = {'a': 'string'}
        c = Config()
        c.set('a.b.c', 42)
        self.assertEqual(c.get('a.b.c'), 42)

    def test_nested_get_set_consistency(self):
        """嵌套get/set应一致"""
        Config._instance = None
        Config._config = {}
        c = Config()
        c.set('a.b.c', 42)
        self.assertEqual(c.get('a.b.c'), 42)
        self.assertEqual(c.get('a.b'), {'c': 42})
        self.assertEqual(c.get('a'), {'b': {'c': 42}})

    def test_missing_key_default(self):
        """缺失键应返回默认值"""
        Config._instance = None
        Config._config = {'a': {'b': 1}}
        c = Config()
        self.assertIsNone(c.get('a.c'))
        self.assertEqual(c.get('a.c', 'default'), 'default')
        self.assertEqual(c.get('nonexistent.path', 123), 123)

    def test_none_value_vs_missing(self):
        """显式None值不应触发默认值"""
        Config._instance = None
        Config._config = {'a': {'b': None}}
        c = Config()
        self.assertIsNone(c.get('a.b', 'default'))

    def test_to_dict_isolation(self):
        """to_dict返回的副本应独立"""
        Config._instance = None
        Config._config = {'a': 1}
        c = Config()
        d = c.to_dict()
        d['a'] = 999
        self.assertEqual(c.get('a'), 1)

    def test_load_replaces_config(self):
        """load应替换整个配置"""
        Config._instance = None
        Config._config = {'old': 'value'}
        c = Config()

        temp_path = os.path.join(tempfile.mkdtemp(), 'config.yaml')
        with open(temp_path, 'w') as f:
            f.write("new: value\n")

        c.load(temp_path)
        self.assertIsNone(c.get('old'))
        self.assertEqual(c.get('new'), 'value')
        shutil.rmtree(os.path.dirname(temp_path), ignore_errors=True)


class TestConfigDefaults(unittest.TestCase):
    """配置默认值测试"""

    def tearDown(self):
        Config._instance = None
        Config._config = {}
        Config._lock = threading.Lock()

    def test_get_base_weights_default(self):
        Config._instance = None
        Config._config = {}
        c = Config()
        self.assertEqual(c.get_base_weights(), (0.4, 0.4, 0.2))

    def test_get_resource_limits_default(self):
        Config._instance = None
        Config._config = {}
        c = Config()
        limits = c.get_resource_limits()
        self.assertEqual(limits['max_cpu_percent'], 20.0)
        self.assertEqual(limits['max_mem_percent'], 20.0)

    def test_get_exploration_config_default(self):
        Config._instance = None
        Config._config = {}
        c = Config()
        cfg = c.get_exploration_config()
        self.assertEqual(cfg['base_ratio'], 0.3)

    def test_get_safety_config_default(self):
        Config._instance = None
        Config._config = {}
        c = Config()
        cfg = c.get_safety_config()
        self.assertEqual(cfg['max_consecutive_failures'], 5)

    def test_get_memory_config_default(self):
        Config._instance = None
        Config._config = {}
        c = Config()
        cfg = c.get_memory_config()
        self.assertEqual(cfg['max_short_term_samples'], 10000)

    def test_get_paths_default(self):
        Config._instance = None
        Config._config = {}
        c = Config()
        paths = c.get_paths()
        self.assertEqual(paths['data_dir'], 'data')


class TestConfigReloaderReliability(unittest.TestCase):
    """ConfigReloader可靠性测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.yaml')
        with open(self.config_path, 'w') as f:
            f.write("key: value1\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_force_reload_returns_config(self):
        """force_reload应返回配置"""
        reloader = ConfigReloader(self.config_path)
        config = reloader.force_reload()
        self.assertIsNotNone(config)
        self.assertEqual(config.get('key'), 'value1')

    def test_force_reload_on_changed_file(self):
        """文件修改后force_reload应返回新配置"""
        reloader = ConfigReloader(self.config_path)
        reloader.force_reload()

        with open(self.config_path, 'w') as f:
            f.write("key: value2\n")

        config = reloader.force_reload()
        self.assertEqual(config.get('key'), 'value2')

    def test_callback_invoked_on_reload(self):
        """回调应在重载时被调用"""
        reloader = ConfigReloader(self.config_path)
        callbacks = []

        def cb(config):
            callbacks.append(config)

        reloader.add_callback(cb)

        with open(self.config_path, 'w') as f:
            f.write("key: new_value\n")

        reloader.force_reload()
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0].get('key'), 'new_value')

    def test_multiple_callbacks(self):
        """多个回调应都被调用"""
        reloader = ConfigReloader(self.config_path)
        count = [0, 0]

        def cb1(_):
            count[0] += 1

        def cb2(_):
            count[1] += 1

        reloader.add_callback(cb1)
        reloader.add_callback(cb2)
        reloader.force_reload()

        self.assertEqual(count[0], 1)
        self.assertEqual(count[1], 1)

    def test_callback_error_does_not_break_others(self):
        """一个回调出错不应影响其他回调"""
        reloader = ConfigReloader(self.config_path)
        success = [False]

        def bad_cb(_):
            raise RuntimeError("bad callback")

        def good_cb(_):
            success[0] = True

        reloader.add_callback(bad_cb)
        reloader.add_callback(good_cb)
        reloader.force_reload()

        self.assertTrue(success[0])

    def test_remove_callback(self):
        """移除回调后不应再被调用"""
        reloader = ConfigReloader(self.config_path)
        count = [0]

        def cb(_):
            count[0] += 1

        reloader.add_callback(cb)
        reloader.force_reload()
        self.assertEqual(count[0], 1)

        reloader.remove_callback(cb)
        reloader.force_reload()
        self.assertEqual(count[0], 1)

    def test_watch_loop_detects_changes(self):
        """watch_loop应能检测文件变化"""
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        callbacks = []

        def cb(config):
            callbacks.append(config)

        reloader.add_callback(cb)
        reloader.start()

        # 修改文件
        time.sleep(0.2)
        with open(self.config_path, 'w') as f:
            f.write("key: changed\n")

        # 等待检测
        time.sleep(0.5)
        reloader.stop()

        self.assertGreaterEqual(len(callbacks), 1)

    def test_stats_reporting(self):
        """统计信息应正确"""
        reloader = ConfigReloader(self.config_path)
        stats = reloader.get_stats()
        self.assertEqual(stats['config_path'], self.config_path)
        self.assertEqual(stats['callback_count'], 0)
        self.assertFalse(stats['running'])

    def test_invalid_yaml_handling(self):
        """无效YAML应优雅处理"""
        with open(self.config_path, 'w') as f:
            f.write("{invalid yaml: [}\n")
        reloader = ConfigReloader(self.config_path)
        result = reloader.force_reload()
        self.assertIsNone(result)

    def test_deleted_file_handling(self):
        """文件删除后应优雅处理"""
        reloader = ConfigReloader(self.config_path)
        os.remove(self.config_path)
        result = reloader.force_reload()
        self.assertIsNone(result)


class TestConfigFileLoading(unittest.TestCase):
    """配置文件加载测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        Config._lock = threading.RLock()

    def tearDown(self):
        Config._instance = None
        Config._config = {}
        Config._lock = threading.RLock()

    def test_load_valid_yaml(self):
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, 'config.yaml')
        with open(path, 'w') as f:
            f.write("a:\n  b:\n    c: 42\nlist_key:\n  - item1\n  - item2\n")
        c = Config(path)
        self.assertEqual(c.get('a.b.c'), 42)
        self.assertEqual(c.get('list_key'), ['item1', 'item2'])
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_empty_file(self):
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, 'empty.yaml')
        with open(path, 'w') as f:
            f.write("")
        c = Config(path)
        self.assertEqual(c.to_dict(), {})
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Config('/nonexistent/path/config.yaml')


if __name__ == '__main__':
    unittest.main()
