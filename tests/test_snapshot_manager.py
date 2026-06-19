import os
import sys
import unittest
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.snapshot_manager import SnapshotManager


class TestSnapshotManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=5)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_snapshot(self):
        sid = self.sm.create_snapshot({'model': 'v1'}, tag='test', metrics={'mae': 0.1})
        self.assertTrue(sid.startswith('test_'))
        self.assertEqual(len(self.sm.list_snapshots()), 1)

    def test_load_snapshot(self):
        state = {'weights': [0.1, 0.2]}
        sid = self.sm.create_snapshot(state, tag='test')
        loaded = self.sm.load_snapshot(sid)
        self.assertEqual(loaded, state)

    def test_load_nonexistent(self):
        result = self.sm.load_snapshot('nonexistent')
        self.assertIsNone(result)

    def test_get_latest(self):
        self.sm.create_snapshot({'v': 1}, tag='auto')
        time.sleep(0.01)
        self.sm.create_snapshot({'v': 2}, tag='auto')
        latest = self.sm.get_latest_snapshot(tag='auto')
        self.assertIsNotNone(latest)

    def test_enforce_limit(self):
        for i in range(7):
            self.sm.create_snapshot({'v': i}, tag='auto')
        self.assertLessEqual(len(self.sm.list_snapshots()), 5)

    def test_delete_snapshot(self):
        sid = self.sm.create_snapshot({'v': 1}, tag='test')
        result = self.sm.delete_snapshot(sid)
        self.assertTrue(result)
        self.assertEqual(len(self.sm.list_snapshots()), 0)

    def test_compare_snapshots(self):
        sid1 = self.sm.create_snapshot({'v': 1}, tag='test', metrics={'mae': 0.2})
        time.sleep(0.01)
        sid2 = self.sm.create_snapshot({'v': 2}, tag='test', metrics={'mae': 0.1})
        result = self.sm.compare_snapshots(sid1, sid2)
        self.assertIn('diffs', result)
        self.assertEqual(result['diffs']['mae'], -0.1)

    def test_index_persistence(self):
        sid = self.sm.create_snapshot({'v': 1}, tag='test')
        sm2 = SnapshotManager(snapshot_dir=self.temp_dir, max_count=5)
        self.assertEqual(len(sm2.list_snapshots()), 1)


if __name__ == '__main__':
    unittest.main()
