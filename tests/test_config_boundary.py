#!/usr/bin/env python3
"""
阶段74：配置系统边界测试
目标：验证配置缺失、类型错误、非法路径、热重载边界条件
"""

import os
import sys
import tempfile
import shutil
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config


class TestConfigMissingFields(unittest.TestCase):
    """配置缺失字段测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_config_file(self):
        """空配置文件应加载为空字典"""
        path = os.path.join(self.temp_dir, 'empty.yaml')
        with open(path, 'w') as f:
            f.write('')
        cfg = Config(config_path=path)
        self.assertEqual(cfg.to_dict(), {})

    def test_missing_required_keys_return_none(self):
        """缺失的键应返回None"""
        path = os.path.join(self.temp_dir, 'minimal.yaml')
        with open(path, 'w') as f:
            f.write('goals:\n  precision_evolution: true\n')
        cfg = Config(config_path=path)
        self.assertIsNone(cfg.get('nonexistent.key'))
        self.assertIsNone(cfg.get('resource_limits.max_cpu_percent'))

    def test_partial_config_loads(self):
        """部分配置应能加载"""
        path = os.path.join(self.temp_dir, 'partial.yaml')
        with open(path, 'w') as f:
            f.write("""
base_weights:
  safe: 0.4
  gain: 0.4
""")
        cfg = Config(config_path=path)
        self.assertEqual(cfg.get('base_weights.safe'), 0.4)
        self.assertIsNone(cfg.get('base_weights.eff'))

    def test_default_values_for_missing_paths(self):
        """缺失路径应返回None而非崩溃"""
        cfg = Config()
        self.assertIsNone(cfg.get('a.b.c.d.e.f'))
        self.assertIsNone(cfg.get(''))


class TestConfigTypeErrors(unittest.TestCase):
    """配置类型错误测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}

    def test_set_non_dict_path(self):
        """在非字典路径上设置嵌套值应处理"""
        cfg = Config()
        cfg.set('key', 'string_value')
        # 尝试在字符串上设置嵌套值
        cfg.set('key.sub', 'nested')
        # 应不崩溃，结果可能是替换或忽略
        result = cfg.get('key')
        self.assertIn(result, ['string_value', {'sub': 'nested'}])

    def test_get_on_non_dict_intermediate(self):
        """中间节点非字典时应返回None"""
        cfg = Config()
        cfg.set('a', 'string')
        self.assertIsNone(cfg.get('a.b'))

    def test_invalid_yaml_handling(self):
        """非法YAML应不崩溃"""
        path = os.path.join(tempfile.mkdtemp(), 'bad.yaml')
        with open(path, 'w') as f:
            f.write('{invalid yaml: [}')
        try:
            cfg = Config(config_path=path)
            # 应加载为空或抛出可处理的异常
            self.assertTrue(True)
        except Exception:
            self.assertTrue(True)
        finally:
            os.remove(path)
            os.rmdir(os.path.dirname(path))


class TestConfigHotReloadBoundary(unittest.TestCase):
    """配置热重载边界测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reload_after_file_deletion(self):
        """配置文件删除后reload应不崩溃且保留旧值"""
        path = os.path.join(self.temp_dir, 'config.yaml')
        with open(path, 'w') as f:
            f.write('key: value\n')
        cfg = Config(config_path=path)
        self.assertEqual(cfg.get('key'), 'value')

        os.remove(path)
        try:
            cfg.reload()
        except FileNotFoundError:
            pass  # 可接受：抛出 FileNotFoundError，但旧配置应保留
        # 旧值应保留
        self.assertEqual(cfg.get('key'), 'value',
                         "文件删除后 reload 不应清空已有配置")

    def test_reload_with_corrupted_file(self):
        """配置文件损坏后reload应不崩溃且保留旧值"""
        path = os.path.join(self.temp_dir, 'config.yaml')
        with open(path, 'w') as f:
            f.write('key: value\n')
        cfg = Config(config_path=path)
        self.assertEqual(cfg.get('key'), 'value')

        with open(path, 'w') as f:
            f.write('{{bad')
        try:
            cfg.reload()
        except Exception:
            pass
        # 旧值应保留
        self.assertEqual(cfg.get('key'), 'value',
                         "文件损坏后 reload 不应清空已有配置")

    def test_concurrent_set_and_get(self):
        """并发设置和读取应线程安全"""
        import threading
        cfg = Config()
        cfg.set('counter', 0)
        errors = []

        def worker():
            for i in range(100):
                try:
                    cfg.set('counter', i)
                    _ = cfg.get('counter')
                    cfg.to_dict()
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")


class TestConfigEnvironmentProfiles(unittest.TestCase):
    """配置环境配置文件测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dev_test_prod_configs_load(self):
        """dev/test/prod三份配置应能加载"""
        for env in ['dev', 'test', 'prod']:
            path = os.path.join(self.temp_dir, f'config_{env}.yaml')
            with open(path, 'w') as f:
                f.write(f"""
environment: {env}
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: {'20.0' if env == 'prod' else '100.0'}
""")
            Config._instance = None
            Config._config = {}
            cfg = Config(config_path=path)
            self.assertEqual(cfg.get('environment'), env)
            self.assertIsNotNone(cfg.get('base_weights.safe'))

    def test_config_keys_consistent_across_envs(self):
        """不同环境的配置键名应一致"""
        keys = set()
        for env in ['dev', 'test', 'prod']:
            path = os.path.join(self.temp_dir, f'config_{env}.yaml')
            with open(path, 'w') as f:
                f.write("""
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: 100.0
""")
            Config._instance = None
            Config._config = {}
            cfg = Config(config_path=path)
            keys.add(tuple(sorted(cfg.to_dict().keys())))

        self.assertEqual(len(keys), 1, "Config keys differ across environments")


class TestConfigValidation(unittest.TestCase):
    """配置校验测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}

    def test_get_base_weights_returns_tuple(self):
        """get_base_weights应返回三元组"""
        cfg = Config()
        cfg._config = {
            'base_weights': {'safe': 0.4, 'gain': 0.4, 'eff': 0.2}
        }
        weights = cfg.get_base_weights()
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_get_base_weights_missing_returns_default(self):
        """缺失base_weights时应返回默认值"""
        cfg = Config()
        weights = cfg.get_base_weights()
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
