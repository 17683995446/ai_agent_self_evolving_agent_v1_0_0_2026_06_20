import os
import sys
import unittest
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.log_sensor import LogSensor


class TestLogSensor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
        with open(self.log_file, 'w') as f:
            f.write("INFO: System started\n")
        self.sensor = LogSensor(log_paths=[self.log_file], window_seconds=60)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_perceive_empty(self):
        data = self.sensor.perceive()
        self.assertIsInstance(data, dict)
        self.assertIn('error_count', data)
        self.assertIn('warn_count', data)
        self.assertIn('error_trend', data)
        self.assertEqual(data['error_count'], 0)
        self.assertEqual(data['warn_count'], 0)

    def test_detect_errors(self):
        with open(self.log_file, 'a') as f:
            f.write("ERROR: Connection failed\n")
            f.write("ERROR: Database timeout\n")
            f.write("INFO: Retrying...\n")

        data = self.sensor.perceive()
        self.assertEqual(data['error_count'], 2)
        self.assertEqual(data['warn_count'], 0)
        self.assertEqual(data['unique_error_types'], 2)

    def test_detect_warnings(self):
        with open(self.log_file, 'a') as f:
            f.write("WARNING: Low disk space\n")
            f.write("WARN: High latency detected\n")

        data = self.sensor.perceive()
        self.assertEqual(data['error_count'], 0)
        self.assertEqual(data['warn_count'], 2)

    def test_mixed_content(self):
        with open(self.log_file, 'a') as f:
            f.write("INFO: Normal operation\n")
            f.write("ERROR: Critical failure\n")
            f.write("WARNING: Deprecated API\n")
            f.write("DEBUG: Trace info\n")
            f.write("FATAL: System crash\n")

        data = self.sensor.perceive()
        self.assertEqual(data['error_count'], 2)
        self.assertEqual(data['warn_count'], 1)

    def test_incremental_read(self):
        with open(self.log_file, 'a') as f:
            f.write("ERROR: First error\n")

        data1 = self.sensor.perceive()
        self.assertEqual(data1['error_count'], 1)

        data2 = self.sensor.perceive()
        self.assertEqual(data2['error_count'], 0)

        with open(self.log_file, 'a') as f:
            f.write("ERROR: Second error\n")

        data3 = self.sensor.perceive()
        self.assertEqual(data3['error_count'], 1)

    def test_error_trend_computation(self):
        now = time.time()
        self.sensor._error_history.append((now - 30, "ERROR: old"))
        self.sensor._error_history.append((now - 20, "ERROR: old"))
        self.sensor._error_history.append((now - 5, "ERROR: new"))
        self.sensor._error_history.append((now - 3, "ERROR: new"))
        self.sensor._error_history.append((now - 1, "ERROR: new"))

        trend = self.sensor._compute_error_trend(now - 60)
        self.assertGreater(trend, 0.0)

    def test_add_remove_log_path(self):
        new_log = os.path.join(self.temp_dir, 'new.log')
        with open(new_log, 'w') as f:
            f.write("INFO: test\n")

        self.sensor.add_log_path(new_log)
        self.assertIn(new_log, self.sensor.log_paths)

        self.sensor.remove_log_path(new_log)
        self.assertNotIn(new_log, self.sensor.log_paths)

    def test_file_rotation(self):
        with open(self.log_file, 'a') as f:
            f.write("ERROR: before rotation\n")

        data1 = self.sensor.perceive()
        self.assertEqual(data1['error_count'], 1)

        with open(self.log_file, 'w') as f:
            f.write("ERROR: after rotation\n")

        data2 = self.sensor.perceive()
        self.assertEqual(data2['error_count'], 1)

    def test_extract_error_type(self):
        line = "2024-01-01 ERROR: Database connection timeout"
        error_type = self.sensor._extract_error_type(line)
        self.assertEqual(error_type, "Database")

    def test_case_insensitive_patterns(self):
        with open(self.log_file, 'a') as f:
            f.write("error: lowercase error\n")
            f.write("Error: Mixed case error\n")
            f.write("ERROR: Uppercase error\n")

        data = self.sensor.perceive()
        self.assertEqual(data['error_count'], 3)


if __name__ == '__main__':
    unittest.main()
