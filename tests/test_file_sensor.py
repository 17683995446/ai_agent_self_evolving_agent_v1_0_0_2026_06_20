import os
import sys
import unittest
import tempfile
import time
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.file_sensor import FileSensor


class TestFileSensor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sensor = FileSensor(watch_paths=[self.temp_dir], window_seconds=60, min_scan_interval=0.0)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_perceive_empty_dir(self):
        data = self.sensor.perceive()
        self.assertIsInstance(data, dict)
        self.assertIn('total_changes', data)
        self.assertIn('new_files', data)
        self.assertIn('deleted_files', data)
        self.assertIn('modified_files', data)
        self.assertEqual(data['total_changes'], 0)
        self.assertEqual(data['total_tracked_files'], 0)

    def test_detect_new_file(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('hello')

        data = self.sensor.perceive()
        self.assertEqual(data['new_files'], 1)
        self.assertEqual(data['total_changes'], 1)
        self.assertEqual(data['total_tracked_files'], 1)

    def test_detect_modified_file(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('hello')

        self.sensor.perceive()

        time.sleep(0.1)
        with open(test_file, 'w') as f:
            f.write('hello world')

        data = self.sensor.perceive()
        self.assertEqual(data['modified_files'], 1)
        self.assertEqual(data['total_changes'], 1)

    def test_detect_deleted_file(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('hello')

        self.sensor.perceive()
        os.remove(test_file)

        data = self.sensor.perceive()
        self.assertEqual(data['deleted_files'], 1)
        self.assertEqual(data['total_changes'], 1)
        self.assertEqual(data['total_tracked_files'], 0)

    def test_detect_large_file(self):
        test_file = os.path.join(self.temp_dir, 'large.bin')
        with open(test_file, 'wb') as f:
            f.write(b'0' * (101 * 1024 * 1024))

        data = self.sensor.perceive()
        self.assertEqual(data['large_files_count'], 1)

    def test_size_delta_calculation(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('a' * 1024)

        self.sensor.perceive()

        time.sleep(0.1)
        with open(test_file, 'w') as f:
            f.write('b' * 2048)

        data = self.sensor.perceive()
        self.assertGreater(data['size_delta_mb'], 0)

    def test_recent_changes_window(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('hello')

        data = self.sensor.perceive()
        self.assertEqual(data['recent_changes'], 1)

    def test_directory_stats(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('a' * 1024)

        data = self.sensor.perceive()
        self.assertIn(self.temp_dir, data['dir_stats'])
        self.assertIn('file_count', data['dir_stats'][self.temp_dir])
        self.assertIn('total_size_mb', data['dir_stats'][self.temp_dir])

    def test_add_remove_watch_path(self):
        new_dir = tempfile.mkdtemp()
        try:
            self.sensor.add_watch_path(new_dir)
            self.assertIn(new_dir, self.sensor.watch_paths)

            test_file = os.path.join(new_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('hello')

            data = self.sensor.perceive()
            self.assertEqual(data['monitored_dirs'], 2)

            self.sensor.remove_watch_path(new_dir)
            self.assertNotIn(new_dir, self.sensor.watch_paths)
        finally:
            shutil.rmtree(new_dir, ignore_errors=True)

    def test_multiple_files(self):
        for i in range(5):
            test_file = os.path.join(self.temp_dir, f'test{i}.txt')
            with open(test_file, 'w') as f:
                f.write(f'content {i}')

        data = self.sensor.perceive()
        self.assertEqual(data['new_files'], 5)
        self.assertEqual(data['total_tracked_files'], 5)

    def test_nested_directories(self):
        nested_dir = os.path.join(self.temp_dir, 'nested', 'deep')
        os.makedirs(nested_dir)

        test_file = os.path.join(nested_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('deep content')

        data = self.sensor.perceive()
        self.assertEqual(data['new_files'], 1)
        self.assertEqual(data['total_tracked_files'], 1)

    def test_permission_error_handling(self):
        restricted_dir = os.path.join(self.temp_dir, 'restricted')
        os.makedirs(restricted_dir)

        test_file = os.path.join(restricted_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('content')

        os.chmod(restricted_dir, 0o000)
        try:
            data = self.sensor.perceive()
            self.assertIsInstance(data, dict)
        finally:
            os.chmod(restricted_dir, 0o755)


if __name__ == '__main__':
    unittest.main()
