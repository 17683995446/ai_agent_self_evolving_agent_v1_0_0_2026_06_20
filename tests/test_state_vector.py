import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.state_vector import StateVector


class TestStateVector(unittest.TestCase):
    def setUp(self):
        self.state_vector = StateVector()

    def test_perceive_returns_dict(self):
        state = self.state_vector.perceive()
        self.assertIsInstance(state, dict)
        self.assertIn('timestamp', state)
        self.assertIn('system', state)
        self.assertIn('logs', state)
        self.assertIn('files', state)
        self.assertIn('self', state)

    def test_system_keys(self):
        state = self.state_vector.perceive()
        system = state['system']
        self.assertIn('cpu_percent', system)
        self.assertIn('cpu_load_1m', system)
        self.assertIn('mem_percent', system)
        self.assertIn('disk_percent', system)
        self.assertIn('process_count', system)
        self.assertIn('zombie_count', system)

    def test_logs_keys(self):
        state = self.state_vector.perceive()
        logs = state['logs']
        self.assertIn('error_count_1m', logs)
        self.assertIn('warn_count_1m', logs)
        self.assertIn('error_trend', logs)

    def test_files_keys(self):
        state = self.state_vector.perceive()
        files = state['files']
        self.assertIn('tmp_size_gb', files)
        self.assertIn('recent_changes_1m', files)
        self.assertIn('large_files_count', files)

    def test_self_keys(self):
        state = self.state_vector.perceive()
        self_data = state['self']
        self.assertIn('self_cpu_percent', self_data)
        self.assertIn('self_mem_mb', self_data)
        self.assertIn('recent_mae', self_data)
        self.assertIn('recent_tasks_success_rate', self_data)
        self.assertIn('coverage_ratio', self_data)

    def test_mae_update(self):
        self.state_vector.update_mae(0.3)
        self.state_vector.update_mae(0.5)
        self.assertEqual(self.state_vector._get_recent_mae(), 0.4)

    def test_task_success_update(self):
        self.state_vector.update_task_success(True)
        self.state_vector.update_task_success(False)
        self.assertEqual(self.state_vector._get_recent_success_rate(), 0.5)

    def test_coverage_update(self):
        self.state_vector.update_coverage(0.8)
        state = self.state_vector.perceive()
        self.assertEqual(state['self']['coverage_ratio'], 0.8)

    def test_coverage_clamping(self):
        self.state_vector.update_coverage(1.5)
        state = self.state_vector.perceive()
        self.assertEqual(state['self']['coverage_ratio'], 1.0)

        self.state_vector.update_coverage(-0.5)
        state = self.state_vector.perceive()
        self.assertEqual(state['self']['coverage_ratio'], 0.0)

    def test_flatten(self):
        state = self.state_vector.perceive()
        flat = self.state_vector.flatten(state)
        self.assertIsInstance(flat, dict)
        for key, value in flat.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)
        self.assertIn('system.cpu_percent', flat)
        self.assertIn('system.mem_percent', flat)

    def test_history_window_limit(self):
        config = {'history_window': 3}
        sv = StateVector(config)
        sv.update_mae(0.1)
        sv.update_mae(0.2)
        sv.update_mae(0.3)
        sv.update_mae(0.4)
        self.assertEqual(len(sv._recent_mae_history), 3)
        self.assertEqual(sv._get_recent_mae(), 0.3)

    def test_default_mae(self):
        self.assertEqual(self.state_vector._get_recent_mae(), 0.5)

    def test_default_success_rate(self):
        self.assertEqual(self.state_vector._get_recent_success_rate(), 0.5)


if __name__ == '__main__':
    unittest.main()
