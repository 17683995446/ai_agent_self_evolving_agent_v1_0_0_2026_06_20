import os
import sys
import unittest
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.short_term import ShortTermMemory


class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_memory.db')
        self.memory = ShortTermMemory(db_path=self.db_path, max_samples=10, batch_size=1, flush_interval=0.0)

    def tearDown(self):
        self.memory.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_tables(self):
        cursor = self.memory._conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        self.assertIn('samples', tables)
        self.assertIn('tasks', tables)
        self.assertIn('model_snapshots', tables)

    def test_store_sample(self):
        sample = {
            'timestamp': time.time(),
            'state': {'cpu': 0.5},
            'predictions': {'cpu': 0.4},
            'actuals': {'cpu': 0.5},
            'errors': {'cpu': 0.1},
            'task_id': 'task_001',
            'reward': 0.8,
            'value_score': 0.5,
        }
        sample_id = self.memory.store_sample(sample)
        self.assertIsInstance(sample_id, int)
        self.assertGreater(sample_id, 0)

    def test_load_samples(self):
        for i in range(5):
            sample = {
                'timestamp': time.time(),
                'state': {'cpu': i * 0.1},
                'predictions': {'cpu': i * 0.1},
                'actuals': {'cpu': i * 0.1},
                'errors': {'cpu': 0.0},
            }
            self.memory.store_sample(sample)

        samples = self.memory.load_samples(limit=3)
        self.assertEqual(len(samples), 3)
        self.assertIn('state_json', samples[0])
        self.assertIsInstance(samples[0]['state_json'], dict)

    def test_load_high_value_samples(self):
        for i in range(5):
            sample = {
                'timestamp': time.time(),
                'state': {},
                'predictions': {},
                'actuals': {},
                'errors': {},
                'value_score': float(i) / 5,
            }
            self.memory.store_sample(sample)

        samples = self.memory.load_high_value_samples(limit=3)
        self.assertEqual(len(samples), 3)
        scores = [s['value_score'] for s in samples]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_update_value_score(self):
        sample = {
            'timestamp': time.time(),
            'state': {},
            'predictions': {},
            'actuals': {},
            'errors': {},
            'value_score': 0.1,
        }
        sample_id = self.memory.store_sample(sample)
        self.memory.update_value_score(sample_id, 0.9)

        samples = self.memory.load_samples(limit=1)
        self.assertEqual(samples[0]['value_score'], 0.9)

    def test_store_task(self):
        task_data = {
            'id': 'task_001',
            'type': 'safe',
            'name': 'cpu_monitor',
            'created_at': time.time(),
            'execution_count': 5,
            'success_count': 4,
            'frozen': False,
            'avg_reward': 0.8,
        }
        self.memory.store_task(task_data)

        cursor = self.memory._conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", ('task_001',))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[1], 'safe')

    def test_store_snapshot(self):
        snapshot_id = self.memory.store_snapshot('/path/to/snap.pkl', {
            'mae': 0.1,
            'coverage': 0.8,
            'eff_score': 0.9,
            'is_stable': True,
        })
        self.assertIsInstance(snapshot_id, int)

    def test_get_latest_snapshot(self):
        self.memory.store_snapshot('/path/old.pkl', {'mae': 0.2})
        time.sleep(0.01)
        self.memory.store_snapshot('/path/new.pkl', {'mae': 0.1})

        latest = self.memory.get_latest_snapshot()
        self.assertIsNotNone(latest)
        self.assertEqual(latest['snapshot_path'], '/path/new.pkl')

    def test_cleanup_old_samples(self):
        for i in range(15):
            sample = {
                'timestamp': time.time(),
                'state': {},
                'predictions': {},
                'actuals': {},
                'errors': {},
                'value_score': float(i) / 15,
            }
            self.memory.store_sample(sample)

        count_before = self.memory.get_sample_count()
        self.assertEqual(count_before, 15)

        deleted = self.memory.cleanup_old_samples(retention_ratio=0.3)
        self.assertGreater(deleted, 0)

        count_after = self.memory.get_sample_count()
        self.assertLess(count_after, count_before)

    def test_get_sample_count(self):
        self.assertEqual(self.memory.get_sample_count(), 0)
        self.memory.store_sample({'timestamp': time.time(), 'state': {}, 'predictions': {}, 'actuals': {}, 'errors': {}})
        self.assertEqual(self.memory.get_sample_count(), 1)

    def test_json_roundtrip(self):
        complex_state = {'nested': {'value': 42, 'list': [1, 2, 3]}}
        sample = {
            'timestamp': time.time(),
            'state': complex_state,
            'predictions': {},
            'actuals': {},
            'errors': {},
        }
        self.memory.store_sample(sample)
        loaded = self.memory.load_samples(limit=1)
        self.assertEqual(loaded[0]['state_json'], complex_state)


if __name__ == '__main__':
    unittest.main()
