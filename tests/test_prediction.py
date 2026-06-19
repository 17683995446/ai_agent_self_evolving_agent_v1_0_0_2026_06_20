import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.base_predictor import ExponentialSmoother, AdaptiveExponentialSmoother, SimpleMovingAverage
from src.prediction.arima_predictor import ARIMAModel
from src.prediction.multi_predictor import MultiPredictor
from src.prediction.error_evaluator import ErrorEvaluator


class TestExponentialSmoother(unittest.TestCase):
    def test_initial_predict_returns_zero(self):
        model = ExponentialSmoother(alpha=0.3)
        preds = model.predict(steps=3)
        self.assertEqual(preds, [0.0, 0.0, 0.0])

    def test_update_and_predict(self):
        model = ExponentialSmoother(alpha=0.5)
        model.update(10.0)
        preds = model.predict(steps=1)
        self.assertEqual(preds[0], 10.0)

    def test_smoothing_effect(self):
        model = ExponentialSmoother(alpha=0.5)
        model.update(10.0)
        model.update(20.0)
        pred = model.predict(steps=1)[0]
        self.assertEqual(pred, 15.0)

    def test_fit_series(self):
        model = ExponentialSmoother(alpha=0.3)
        model.fit([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = model.predict(steps=1)[0]
        self.assertGreater(pred, 0.0)

    def test_get_set_params(self):
        model = ExponentialSmoother(alpha=0.3)
        model.update(10.0)
        params = model.get_params()
        self.assertEqual(params['alpha'], 0.3)
        self.assertEqual(params['level'], 10.0)

        new_model = ExponentialSmoother()
        new_model.set_params(params)
        self.assertEqual(new_model.alpha, 0.3)
        self.assertEqual(new_model.level, 10.0)


class TestAdaptiveExponentialSmoother(unittest.TestCase):
    def test_alpha_adaptation_increase(self):
        model = AdaptiveExponentialSmoother(alpha_init=0.3, adaptation_rate=0.1)
        model.update(10.0)
        model.update(20.0)
        self.assertGreater(model.alpha, 0.3)

    def test_alpha_adaptation_decrease(self):
        model = AdaptiveExponentialSmoother(alpha_init=0.5, adaptation_rate=0.1)
        model.update(10.0)
        model.update(10.0)
        model.update(10.0)
        self.assertLess(model.alpha, 0.5)

    def test_alpha_bounds(self):
        model = AdaptiveExponentialSmoother(alpha_init=0.3, alpha_min=0.05, alpha_max=0.8)
        for _ in range(100):
            model.update(10.0)
            model.update(100.0)
        self.assertLessEqual(model.alpha, 0.8)


class TestSimpleMovingAverage(unittest.TestCase):
    def test_empty_predict(self):
        model = SimpleMovingAverage(window=5)
        preds = model.predict(steps=1)
        self.assertEqual(preds[0], 0.0)

    def test_window_average(self):
        model = SimpleMovingAverage(window=3)
        model.update(10.0)
        model.update(20.0)
        model.update(30.0)
        pred = model.predict(steps=1)[0]
        self.assertEqual(pred, 20.0)

    def test_window_sliding(self):
        model = SimpleMovingAverage(window=3)
        for i in range(5):
            model.update(float(i * 10))
        pred = model.predict(steps=1)[0]
        self.assertEqual(pred, 30.0)


class TestARIMAModel(unittest.TestCase):
    def test_insufficient_samples(self):
        model = ARIMAModel(order=(1, 1, 1), min_samples=10)
        model.update(1.0)
        model.update(2.0)
        preds = model.predict(steps=1)
        self.assertEqual(len(preds), 1)

    def test_enough_samples(self):
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        for i in range(10):
            model.update(float(i))
        preds = model.predict(steps=1)
        self.assertEqual(len(preds), 1)


class TestMultiPredictor(unittest.TestCase):
    def test_init_default_metrics(self):
        predictor = MultiPredictor()
        self.assertGreater(len(predictor.metrics), 0)
        self.assertEqual(len(predictor.predictors), len(predictor.metrics))

    def test_predict_all(self):
        predictor = MultiPredictor(metrics=['a', 'b'])
        preds = predictor.predict_all()
        self.assertIn('a', preds)
        self.assertIn('b', preds)
        self.assertEqual(preds['a'], 0.0)

    def test_update_and_predict(self):
        predictor = MultiPredictor(metrics=['a'])
        predictor.update_all({'a': 10.0})
        preds = predictor.predict_all()
        self.assertEqual(preds['a'], 10.0)

    def test_add_remove_metric(self):
        predictor = MultiPredictor()
        predictor.add_metric('new_metric')
        self.assertIn('new_metric', predictor.metrics)
        self.assertIn('new_metric', predictor.predictors)

        predictor.remove_metric('new_metric')
        self.assertNotIn('new_metric', predictor.metrics)
        self.assertNotIn('new_metric', predictor.predictors)

    def test_model_type_selection(self):
        predictor = MultiPredictor(model_type='sma', model_config={'window': 5})
        self.assertEqual(predictor.model_type, 'sma')


class TestErrorEvaluator(unittest.TestCase):
    def test_evaluate(self):
        evaluator = ErrorEvaluator()
        predictions = {'a': 10.0, 'b': 20.0}
        actuals = {'a': 12.0, 'b': 18.0}
        result = evaluator.evaluate(predictions, actuals)
        self.assertEqual(result['mae'], 2.0)
        self.assertEqual(result['mse'], 4.0)
        self.assertEqual(result['max_error'], 2.0)

    def test_evaluate_missing_actual(self):
        evaluator = ErrorEvaluator()
        predictions = {'a': 10.0}
        actuals = {}
        result = evaluator.evaluate(predictions, actuals)
        self.assertEqual(result['mae'], 10.0)

    def test_evaluate_single(self):
        evaluator = ErrorEvaluator()
        result = evaluator.evaluate_single(10.0, 12.0)
        self.assertEqual(result['error'], 2.0)
        self.assertEqual(result['squared_error'], 4.0)


if __name__ == '__main__':
    unittest.main()
