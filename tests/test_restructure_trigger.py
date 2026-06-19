import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.restructure_trigger import RestructureTrigger


class TestRestructureTrigger(unittest.TestCase):
    def test_high_error_triggers(self):
        t = RestructureTrigger()
        self.assertTrue(t.should_restructure(0.6, []))

    def test_low_error_no_trigger(self):
        t = RestructureTrigger()
        self.assertFalse(t.should_restructure(0.1, []))

    def test_max_limit_blocks(self):
        t = RestructureTrigger(max_restructures=2)
        t.record_restructure()
        t.record_restructure()
        self.assertFalse(t.should_restructure(0.9, []))

    def test_stable_window_trigger(self):
        t = RestructureTrigger(stable_window=5)
        errors = [0.1] * 5 + [0.1] * 5
        self.assertTrue(t.should_restructure(0.1, errors))

    def test_record_increments(self):
        t = RestructureTrigger()
        t.record_restructure()
        self.assertEqual(t._restructure_count, 1)
        self.assertIsNotNone(t._last_restructure_time)

    def test_get_stats(self):
        t = RestructureTrigger(max_restructures=3)
        t.record_restructure()
        stats = t.get_stats()
        self.assertEqual(stats['restructure_count'], 1)
        self.assertEqual(stats['remaining'], 2)

    def test_reset(self):
        t = RestructureTrigger()
        t.record_restructure()
        t.reset()
        self.assertEqual(t._restructure_count, 0)
        self.assertIsNone(t._last_restructure_time)


if __name__ == '__main__':
    unittest.main()
