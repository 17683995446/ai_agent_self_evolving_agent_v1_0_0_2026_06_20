import os
import sys
import unittest
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config_reloader import ConfigReloader


class TestConfigReloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.yaml')

        with open(self.config_path, 'w') as f:
            f.write("key1: value1\nkey2: 42\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        reloader = ConfigReloader(self.config_path)
        self.assertEqual(reloader.config_path, self.config_path)
        self.assertFalse(reloader.is_running())

    def test_detect_change(self):
        reloader = ConfigReloader(self.config_path)
        # 初始化后状态已记录
        self.assertFalse(reloader._update_file_state())

        # 修改文件
        time.sleep(0.1)
        with open(self.config_path, 'w') as f:
            f.write("key1: changed\nkey2: 42\n")

        self.assertTrue(reloader._update_file_state())

    def test_callback_triggered(self):
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        triggered = []

        def callback(config):
            triggered.append(config)

        reloader.add_callback(callback)
        reloader.start()

        # 修改文件
        time.sleep(0.15)
        with open(self.config_path, 'w') as f:
            f.write("key1: changed\n")

        time.sleep(0.3)
        reloader.stop()

        self.assertGreaterEqual(len(triggered), 1)
        self.assertEqual(triggered[0]['key1'], 'changed')

    def test_remove_callback(self):
        reloader = ConfigReloader(self.config_path)

        def callback(config):
            pass

        reloader.add_callback(callback)
        self.assertTrue(reloader.remove_callback(callback))
        self.assertFalse(reloader.remove_callback(callback))

    def test_force_reload(self):
        reloader = ConfigReloader(self.config_path)

        with open(self.config_path, 'w') as f:
            f.write("forced: true\n")

        result = reloader.force_reload()
        self.assertIsNotNone(result)
        self.assertEqual(result['forced'], True)

    def test_no_change_no_trigger(self):
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        triggered = []

        def callback(config):
            triggered.append(config)

        reloader.add_callback(callback)
        reloader.start()

        # 不修改文件，等待
        time.sleep(0.3)
        reloader.stop()

        # 初始化时不会触发回调（因为文件没有变化）
        self.assertEqual(len(triggered), 0)

    def test_stats(self):
        reloader = ConfigReloader(self.config_path)
        stats = reloader.get_stats()
        self.assertEqual(stats['config_path'], self.config_path)
        self.assertEqual(stats['check_interval'], 5.0)
        self.assertEqual(stats['callback_count'], 0)

    def test_multiple_callbacks(self):
        reloader = ConfigReloader(self.config_path, check_interval=0.1)
        results = []

        def cb1(config):
            results.append('cb1')

        def cb2(config):
            results.append('cb2')

        reloader.add_callback(cb1)
        reloader.add_callback(cb2)
        reloader.start()

        time.sleep(0.15)
        with open(self.config_path, 'w') as f:
            f.write("multi: true\n")

        time.sleep(0.3)
        reloader.stop()

        self.assertIn('cb1', results)
        self.assertIn('cb2', results)

    def test_invalid_config_file(self):
        reloader = ConfigReloader(self.config_path)
        # 指向不存在的文件
        reloader.config_path = '/nonexistent/config.yaml'
        result = reloader._update_file_state()
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
