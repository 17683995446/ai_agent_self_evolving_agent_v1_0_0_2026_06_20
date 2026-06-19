import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config
from src.core.constants import DEFAULT_CONFIG_PATH


class TestConfig(unittest.TestCase):
    def setUp(self):
        Config._instance = None
        Config._config = {}

    def test_load_valid_config(self):
        config_content = """
goals:
  precision_evolution: true
base_weights:
  safe: 0.5
  gain: 0.3
  eff: 0.2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            path = f.name
        try:
            cfg = Config(path)
            self.assertTrue(cfg.get('goals.precision_evolution'))
            self.assertEqual(cfg.get('base_weights.safe'), 0.5)
        finally:
            os.unlink(path)

    def test_get_nested_key(self):
        Config._config = {'a': {'b': {'c': 42}}}
        cfg = Config()
        self.assertEqual(cfg.get('a.b.c'), 42)
        self.assertIsNone(cfg.get('a.b.x'))
        self.assertEqual(cfg.get('a.b.x', 'default'), 'default')

    def test_set_nested_key(self):
        Config._config = {}
        cfg = Config()
        cfg.set('a.b.c', 100)
        self.assertEqual(cfg.get('a.b.c'), 100)

    def test_get_base_weights(self):
        Config._config = {'base_weights': {'safe': 0.6, 'gain': 0.2, 'eff': 0.2}}
        cfg = Config()
        weights = cfg.get_base_weights()
        self.assertEqual(weights, (0.6, 0.2, 0.2))

    def test_get_base_weights_default(self):
        Config._config = {}
        cfg = Config()
        weights = cfg.get_base_weights()
        self.assertEqual(weights, (0.4, 0.4, 0.2))

    def test_ensure_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Config._config = {'paths': {'data_dir': os.path.join(tmpdir, 'data')}}
            cfg = Config()
            cfg.ensure_paths()
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'data')))

    def test_singleton(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value1\n")
            path = f.name
        try:
            cfg1 = Config(path)
            cfg2 = Config()
            self.assertIs(cfg1, cfg2)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
