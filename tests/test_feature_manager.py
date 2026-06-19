import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.feature_manager import FeatureManager, ModelReassembler


class TestFeatureManager(unittest.TestCase):
    def test_init(self):
        fm = FeatureManager(['cpu', 'mem'])
        self.assertEqual(fm.get_active_features(), ['cpu', 'mem'])

    def test_add_candidate(self):
        fm = FeatureManager(['cpu'])
        fm.add_candidate('disk')
        self.assertIn('disk', fm.get_candidate_features())

    def test_promote_good_feature(self):
        fm = FeatureManager(['cpu'])
        fm.add_candidate('disk')
        for _ in range(10):
            promoted = fm.evaluate_and_promote({'disk': 0.1})
        self.assertIn('disk', fm.get_active_features())

    def test_no_promote_bad_feature(self):
        fm = FeatureManager(['cpu'])
        fm.add_candidate('disk')
        for _ in range(10):
            promoted = fm.evaluate_and_promote({'disk': 0.9})
        self.assertNotIn('disk', fm.get_active_features())

    def test_demote_underperforming(self):
        fm = FeatureManager(['cpu', 'mem'])
        demoted = fm.demote_underperforming({'cpu': 0.9}, threshold=0.8)
        self.assertIn('cpu', demoted)
        self.assertNotIn('cpu', fm.get_active_features())


class TestModelReassembler(unittest.TestCase):
    def test_reassemble(self):
        registry = {'cpu': 'predictor_cpu'}
        ra = ModelReassembler(registry)
        assembly = ra.reassemble(['cpu', 'mem'])
        self.assertIn('cpu', assembly)
        self.assertIn('mem', assembly)

    def test_stats(self):
        ra = ModelReassembler({})
        ra.reassemble(['a', 'b'])
        stats = ra.get_assembly_stats()
        self.assertEqual(stats['history_count'], 1)
        self.assertEqual(stats['latest_feature_count'], 2)


if __name__ == '__main__':
    unittest.main()
