import os
import sys
import unittest
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.review_engine import ReviewEngine
from src.memory.short_term import ShortTermMemory
from src.optimization.micro_optimizer import MicroOptimizer


class TestReviewEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_memory.db')
        self.memory = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        self.optimizer = MicroOptimizer()
        self.engine = ReviewEngine(self.memory, self.optimizer, min_samples=5)

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_review_insufficient_samples(self):
        result = self.engine.should_review(sample_count=3, idle_cpu=15.0)
        self.assertFalse(result)

    def test_should_review_low_cpu(self):
        result = self.engine.should_review(sample_count=10, idle_cpu=5.0)
        self.assertFalse(result)

    def test_should_review_ready(self):
        result = self.engine.should_review(sample_count=10, idle_cpu=15.0)
        self.assertTrue(result)

    def test_run_review_no_samples(self):
        result = self.engine.run_review()
        self.assertEqual(result['status'], 'no_samples')
        self.assertEqual(result['processed'], 0)

    def test_run_review_with_samples(self):
        for i in range(10):
            sample = {
                'timestamp': time.time(),
                'state': {'cpu': i * 0.1},
                'predictions': {'cpu': i * 0.09},
                'actuals': {'cpu': i * 0.1},
                'errors': {'cpu': 0.01},
                'task_id': 'task_001',
                'reward': 0.8,
                'value_score': 0.5,
            }
            self.memory.store_sample(sample)

        result = self.engine.run_review()
        self.assertEqual(result['status'], 'completed')
        self.assertGreater(result['processed'], 0)
        self.assertGreaterEqual(result['replayed'], 0)

    def test_generate_report(self):
        review_result = {
            'status': 'completed',
            'processed': 100,
            'replayed': 80,
            'duration_seconds': 1.5,
        }
        report = self.engine.generate_report(review_result)
        self.assertIn('Offline Review Report', report)
        self.assertIn('completed', report)
        self.assertIn('100', report)
        self.assertIn('1.50s', report)

    def test_batch_update_value_scores(self):
        for i in range(5):
            sample = {
                'timestamp': time.time(),
                'state': {},
                'predictions': {},
                'actuals': {},
                'errors': {'cpu': i * 0.1},
                'task_id': 'task_001',
            }
            self.memory.store_sample(sample)

        samples = self.memory.load_samples(limit=5)
        updated = self.engine._batch_update_value_scores(samples)
        self.assertGreater(updated, 0)

    def test_replay_samples(self):
        samples = [
            {'error_json': {'cpu': 0.1}},
            {'error_json': {'cpu': 0.2}},
            {'error_json': {}},
        ]
        replayed = self.engine._replay_samples(samples)
        self.assertEqual(replayed, 2)


if __name__ == '__main__':
    unittest.main()
