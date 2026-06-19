import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.micro_optimizer import MicroOptimizer
from src.prediction.base_predictor import AdaptiveExponentialSmoother
from src.prediction.multi_predictor import MultiPredictor


class TestMicroOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = MicroOptimizer(learning_rate=0.1, min_improvement=0.001)
        self.predictor = MultiPredictor(metrics=['cpu', 'mem'])

    def test_optimize_no_history(self):
        errors = {'cpu': 0.5}
        improvements = self.optimizer.optimize(self.predictor, errors)
        self.assertEqual(improvements, {})

    def test_optimize_with_history(self):
        self.optimizer.optimize(self.predictor, {'cpu': 0.5})
        improvements = self.optimizer.optimize(self.predictor, {'cpu': 0.6})
        self.assertIn('cpu', improvements)

    def test_adjust_params(self):
        params = {'alpha': 0.3, 'level': 10.0}
        new_params = self.optimizer._adjust_params(params, 0.6, 0.5)
        self.assertIn('alpha', new_params)
        self.assertGreaterEqual(new_params['alpha'], 0.05)
        self.assertLessEqual(new_params['alpha'], 0.8)

    def test_optimize_adaptive_smoother(self):
        smoother = AdaptiveExponentialSmoother(alpha_init=0.3, adaptation_rate=0.1)
        smoother.update(10.0)
        smoother.update(20.0)
        old_alpha = smoother.alpha
        self.optimizer.optimize_adaptive_smoother(smoother, 30.0)
        self.assertNotEqual(smoother.alpha, old_alpha)

    def test_reset_metric_history(self):
        self.optimizer.optimize(self.predictor, {'cpu': 0.5})
        self.optimizer.optimize(self.predictor, {'cpu': 0.6})
        self.assertEqual(len(self.optimizer._error_history['cpu']), 2)

        self.optimizer.reset_metric_history('cpu')
        self.assertEqual(len(self.optimizer._error_history['cpu']), 0)

    def test_get_metric_trend(self):
        for i in range(10):
            self.optimizer.optimize(self.predictor, {'cpu': float(i) * 0.1})
        trend = self.optimizer.get_metric_trend('cpu')
        self.assertIsNotNone(trend)
        self.assertIsInstance(trend, float)

    def test_get_metric_trend_insufficient(self):
        trend = self.optimizer.get_metric_trend('cpu')
        self.assertIsNone(trend)


if __name__ == '__main__':
    unittest.main()
