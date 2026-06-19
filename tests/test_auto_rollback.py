import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.auto_rollback import AutoRollback
from src.optimization.snapshot_manager import SnapshotManager


class TestAutoRollback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=5)
        self.ar = AutoRollback(self.sm, mae_degradation_threshold=0.2)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_baseline(self):
        result = self.ar.check_and_rollback(0.5, lambda x: True)
        self.assertEqual(result['action'], 'no_baseline')

    def test_continue_when_improved(self):
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.5})
        self.ar.set_baseline(0.5, sid)
        result = self.ar.check_and_rollback(0.3, lambda x: True)
        self.assertEqual(result['action'], 'continue')

    def test_rollback_when_degraded(self):
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.5})
        self.ar.set_baseline(0.5, sid)
        restored = []
        def restore(state):
            restored.append(state)
            return True
        # 退化窗口需3次连续退化才触发回滚
        self.ar.check_and_rollback(0.7, restore)
        self.ar.check_and_rollback(0.7, restore)
        result = self.ar.check_and_rollback(0.7, restore)
        self.assertEqual(result['action'], 'rollback')
        self.assertTrue(result['success'])
        self.assertEqual(len(restored), 1)

    def test_rollback_failed_callback(self):
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.5})
        self.ar.set_baseline(0.5, sid)
        # 退化窗口需3次连续退化才触发回滚
        self.ar.check_and_rollback(0.7, lambda x: True)
        self.ar.check_and_rollback(0.7, lambda x: True)
        result = self.ar.check_and_rollback(0.7, lambda x: False)
        self.assertEqual(result['action'], 'rollback_failed')

    def test_rollback_count(self):
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.5})
        self.ar.set_baseline(0.5, sid)
        # 第一次回滚：需3次连续退化
        self.ar.check_and_rollback(0.7, lambda x: True)
        self.ar.check_and_rollback(0.7, lambda x: True)
        self.ar.check_and_rollback(0.7, lambda x: True)
        # 第二次回滚：再3次连续退化
        self.ar.check_and_rollback(0.7, lambda x: True)
        self.ar.check_and_rollback(0.7, lambda x: True)
        self.ar.check_and_rollback(0.7, lambda x: True)
        self.assertEqual(self.ar._rollback_count, 2)

    def test_get_stats(self):
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.5})
        self.ar.set_baseline(0.5, sid)
        stats = self.ar.get_stats()
        self.assertEqual(stats['baseline_mae'], 0.5)
        self.assertEqual(stats['threshold'], 0.2)


if __name__ == '__main__':
    unittest.main()
