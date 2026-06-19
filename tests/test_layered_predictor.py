import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.layered_predictor import LowLevelPredictor, MidLevelPredictor, HighLevelPredictor


class TestLowLevelPredictor(unittest.TestCase):
    def test_init(self):
        p = LowLevelPredictor(metrics=['cpu', 'mem'])
        self.assertEqual(len(p.predictors), 2)

    def test_predict_before_update(self):
        p = LowLevelPredictor(metrics=['cpu'])
        preds = p.predict()
        self.assertEqual(preds['cpu'], 0.0)

    def test_update_and_predict(self):
        p = LowLevelPredictor(metrics=['cpu'])
        p.update({'cpu': 0.5})
        preds = p.predict()
        self.assertEqual(preds['cpu'], 0.5)

    def test_history(self):
        p = LowLevelPredictor(metrics=['cpu'])
        for i in range(5):
            p.update({'cpu': i * 0.1})
        hist = p.get_history('cpu')
        self.assertEqual(len(hist), 5)
        self.assertEqual(hist[-1], 0.4)


class TestMidLevelPredictor(unittest.TestCase):
    def test_init(self):
        p = MidLevelPredictor(metrics=['cpu'])
        self.assertEqual(len(p.predictors), 1)

    def test_predict_insufficient_samples(self):
        p = MidLevelPredictor(metrics=['cpu'])
        p.update({'cpu': 0.5})
        preds = p.predict()
        self.assertEqual(preds['cpu'], 0.5)

    def test_trend_computation(self):
        p = MidLevelPredictor(metrics=['cpu'])
        for i in range(5):
            p.update({'cpu': i * 0.1})
        trend = p.get_trend('cpu')
        self.assertIsNotNone(trend)
        self.assertGreater(trend, 0.0)


class TestHighLevelPredictor(unittest.TestCase):
    def test_predict(self):
        p = HighLevelPredictor()
        result = p.predict({'cpu': 0.3}, {'cpu': 0.5})
        self.assertIn('combined_predictions', result)
        self.assertIn('anomaly_likelihood', result)
        self.assertIn('recommended_action', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['combined_predictions']['cpu'], 0.4)

    def test_anomaly_high(self):
        p = HighLevelPredictor()
        result = p.predict({'cpu': 0.9}, {'cpu': 0.9})
        self.assertGreater(result['anomaly_likelihood'], 0.7)
        self.assertEqual(result['recommended_action'], 'emergency_safeguard')

    def test_anomaly_low(self):
        p = HighLevelPredictor()
        result = p.predict({'cpu': 0.1}, {'cpu': 0.1})
        self.assertLess(result['anomaly_likelihood'], 0.4)
        self.assertEqual(result['recommended_action'], 'continue_exploration')

    def test_update_and_best_strategy(self):
        p = HighLevelPredictor()
        p.update({'system': {'cpu_percent': 0.5}}, 'explore', 0.9)
        p.update({'system': {'cpu_percent': 0.5}}, 'exploit', 0.3)
        best = p.get_best_strategy()
        self.assertEqual(best, 'explore')

    def test_state_summary(self):
        p = HighLevelPredictor()
        # 使用真实百分比值测试（cpu_percent/mem_percent范围0-100）
        self.assertEqual(p._summarize_state({'system': {'cpu_percent': 90.0, 'mem_percent': 20.0}}), 'high_load')
        self.assertEqual(p._summarize_state({'system': {'cpu_percent': 10.0, 'mem_percent': 20.0}}), 'low_load')
        self.assertEqual(p._summarize_state({'system': {'cpu_percent': 50.0, 'mem_percent': 50.0}}), 'normal')


if __name__ == '__main__':
    unittest.main()
