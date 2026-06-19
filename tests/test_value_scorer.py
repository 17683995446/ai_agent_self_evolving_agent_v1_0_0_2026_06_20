import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.value_scorer import ValueScorer


class TestValueScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ValueScorer()

    def test_high_error_high_score(self):
        sample = {
            'timestamp': time.time(),
            'mae': 0.9,
            'errors': {'cpu': 0.9},
            'task_id': 'task_001',
        }
        score = self.scorer.compute_score(sample)
        self.assertGreater(score, 0.5)

    def test_low_error_low_score(self):
        sample = {
            'timestamp': time.time(),
            'mae': 0.01,
            'errors': {'cpu': 0.01},
            'task_id': 'task_001',
        }
        score = self.scorer.compute_score(sample)
        self.assertLess(score, 0.5)

    def test_time_decay(self):
        now = time.time()
        old_sample = {
            'timestamp': now - 200 * 3600,
            'mae': 0.9,
            'errors': {'cpu': 0.9},
            'task_id': 'task_001',
        }
        new_sample = {
            'timestamp': now,
            'mae': 0.9,
            'errors': {'cpu': 0.9},
            'task_id': 'task_001',
        }
        old_score = self.scorer.compute_score(old_sample)
        new_score = self.scorer.compute_score(new_sample)
        self.assertLess(old_score, new_score)

    def test_no_task_id(self):
        sample = {
            'timestamp': time.time(),
            'mae': 0.5,
            'errors': {},
        }
        score = self.scorer.compute_score(sample)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_range(self):
        sample = {
            'timestamp': time.time(),
            'mae': 0.5,
            'errors': {'cpu': 0.5},
            'task_id': 'task_001',
        }
        score = self.scorer.compute_score(sample)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_batch_score(self):
        samples = [
            {'id': 1, 'timestamp': time.time(), 'mae': 0.9, 'errors': {'cpu': 0.9}, 'task_id': 't1'},
            {'id': 2, 'timestamp': time.time(), 'mae': 0.1, 'errors': {'cpu': 0.1}, 'task_id': 't2'},
        ]
        scores = self.scorer.batch_score(samples)
        self.assertEqual(len(scores), 2)
        self.assertIn(1, scores)
        self.assertIn(2, scores)
        self.assertGreater(scores[1], scores[2])

    def test_novelty_from_errors(self):
        sample_high = {
            'timestamp': time.time(),
            'mae': 0.5,
            'errors': {'cpu': 0.9, 'mem': 0.8},
            'task_id': 'task_001',
        }
        sample_low = {
            'timestamp': time.time(),
            'mae': 0.5,
            'errors': {'cpu': 0.1, 'mem': 0.1},
            'task_id': 'task_002',
        }
        score_high = self.scorer.compute_score(sample_high)
        score_low = self.scorer.compute_score(sample_low)
        self.assertGreater(score_high, score_low)


if __name__ == '__main__':
    unittest.main()
