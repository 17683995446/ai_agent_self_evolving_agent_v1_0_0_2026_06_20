import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.system_sensor import SystemSensor


class TestSystemSensor(unittest.TestCase):
    def setUp(self):
        self.sensor = SystemSensor()

    def test_perceive_returns_dict(self):
        data = self.sensor.perceive()
        self.assertIsInstance(data, dict)
        self.assertIn('timestamp', data)
        self.assertIn('cpu', data)
        self.assertIn('memory', data)
        self.assertIn('disk', data)
        self.assertIn('network', data)
        self.assertIn('processes', data)
        self.assertIn('load', data)

    def test_cpu_structure(self):
        data = self.sensor._read_cpu()
        self.assertIn('percent', data)
        self.assertIn('percpu', data)
        self.assertIn('count', data)
        self.assertIsInstance(data['percent'], float)
        self.assertIsInstance(data['percpu'], list)
        self.assertGreaterEqual(data['percent'], 0.0)
        self.assertLessEqual(data['percent'], 1.0)

    def test_memory_structure(self):
        data = self.sensor._read_memory()
        self.assertIn('total_gb', data)
        self.assertIn('available_gb', data)
        self.assertIn('percent', data)
        self.assertIn('used_gb', data)
        self.assertIn('swap_percent', data)
        self.assertGreaterEqual(data['percent'], 0.0)
        self.assertLessEqual(data['percent'], 1.0)

    def test_disk_structure(self):
        data = self.sensor._read_disk()
        self.assertIn('partitions', data)
        self.assertIn('read_mb_s', data)
        self.assertIn('write_mb_s', data)
        self.assertIsInstance(data['partitions'], dict)
        self.assertGreaterEqual(data['read_mb_s'], 0.0)
        self.assertGreaterEqual(data['write_mb_s'], 0.0)

    def test_network_structure(self):
        data = self.sensor._read_network()
        self.assertIn('sent_mb_s', data)
        self.assertIn('recv_mb_s', data)
        self.assertIn('connections', data)
        self.assertGreaterEqual(data['sent_mb_s'], 0.0)
        self.assertGreaterEqual(data['recv_mb_s'], 0.0)
        self.assertIsInstance(data['connections'], int)

    def test_processes_structure(self):
        data = self.sensor._read_processes()
        self.assertIn('total_count', data)
        self.assertIn('zombie_count', data)
        self.assertIn('top_cpu', data)
        self.assertIn('top_mem', data)
        self.assertIsInstance(data['total_count'], int)
        self.assertIsInstance(data['zombie_count'], int)
        self.assertLessEqual(len(data['top_cpu']), 5)
        self.assertLessEqual(len(data['top_mem']), 5)

    def test_load_structure(self):
        data = self.sensor._read_load()
        self.assertIn('load1', data)
        self.assertIn('load5', data)
        self.assertIn('load15', data)
        self.assertIn('load1_per_cpu', data)
        self.assertIsInstance(data['load1'], float)

    def test_io_rate_computation(self):
        class FakeDiskIO:
            read_bytes = 1024 * 1024
            write_bytes = 512 * 1024
        rates = self.sensor._compute_io_rates(FakeDiskIO())
        self.assertIn('read_mb_s', rates)
        self.assertIn('write_mb_s', rates)

    def test_net_rate_computation(self):
        rates = self.sensor._compute_net_rates(1024 * 1024, 512 * 1024)
        self.assertIn('sent_mb_s', rates)
        self.assertIn('recv_mb_s', rates)


if __name__ == '__main__':
    unittest.main()
