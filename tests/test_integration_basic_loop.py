import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.state_vector import StateVector
from src.drive.reward_engine import RewardEngine
from src.prediction.multi_predictor import MultiPredictor
from src.prediction.error_evaluator import ErrorEvaluator
from src.optimization.micro_optimizer import MicroOptimizer


class TestBasicSelfDrivingLoop(unittest.TestCase):
    def setUp(self):
        self.state_vector = StateVector()
        self.reward_engine = RewardEngine(base_weights=(0.4, 0.4, 0.2))
        self.predictor = MultiPredictor(
            metrics=['system.cpu_percent', 'system.mem_percent'],
            model_type='adaptive_exponential'
        )
        self.error_evaluator = ErrorEvaluator()
        self.optimizer = MicroOptimizer()
        self.recent_errors = []

    def test_full_loop_single_iteration(self):
        state = self.state_vector.perceive()
        self.assertIsInstance(state, dict)

        reward_info = self.reward_engine.compute_total_reward(state, self.recent_errors)
        self.assertIn('total_reward', reward_info)
        self.assertIn('safe_gap', reward_info)
        self.assertIn('gain_gap', reward_info)
        self.assertIn('eff_gap', reward_info)

        predictions = self.predictor.predict_all()
        self.assertIn('system.cpu_percent', predictions)
        self.assertIn('system.mem_percent', predictions)

        time.sleep(0.5)
        next_state = self.state_vector.perceive()
        actuals = {
            'system.cpu_percent': next_state['system']['cpu_percent'],
            'system.mem_percent': next_state['system']['mem_percent'],
        }

        error_result = self.error_evaluator.evaluate(predictions, actuals)
        self.assertIn('mae', error_result)
        self.assertIn('errors', error_result)

        self.recent_errors.append(error_result['mae'])
        self.state_vector.update_mae(error_result['mae'])

        self.predictor.update_all(actuals)

        improvements = self.optimizer.optimize(self.predictor, error_result['errors'])
        self.assertIsInstance(improvements, dict)

    def test_multiple_iterations(self):
        for i in range(5):
            state = self.state_vector.perceive()
            reward_info = self.reward_engine.compute_total_reward(state, self.recent_errors)
            predictions = self.predictor.predict_all()

            time.sleep(0.2)
            next_state = self.state_vector.perceive()
            actuals = {
                'system.cpu_percent': next_state['system']['cpu_percent'],
                'system.mem_percent': next_state['system']['mem_percent'],
            }

            error_result = self.error_evaluator.evaluate(predictions, actuals)
            self.recent_errors.append(error_result['mae'])
            self.state_vector.update_mae(error_result['mae'])
            self.predictor.update_all(actuals)
            self.optimizer.optimize(self.predictor, error_result['errors'])

        self.assertEqual(len(self.recent_errors), 5)

    def test_reward_drives_gap(self):
        safe_state = {
            'system': {'cpu_percent': 0.2, 'mem_percent': 0.3, 'disk_percent': 0.4, 'zombie_count': 0},
            'logs': {'error_count_1m': 0},
            'self': {'self_cpu_percent': 2.0, 'self_mem_mb': 64, 'coverage_ratio': 0.8},
        }
        risky_state = {
            'system': {'cpu_percent': 0.95, 'mem_percent': 0.9, 'disk_percent': 0.5, 'zombie_count': 10},
            'logs': {'error_count_1m': 20},
            'self': {'self_cpu_percent': 2.0, 'self_mem_mb': 64, 'coverage_ratio': 0.8},
        }

        safe_reward = self.reward_engine.compute_total_reward(safe_state, [0.1])
        risky_reward = self.reward_engine.compute_total_reward(risky_state, [0.1])

        self.assertGreater(safe_reward['safe_reward'], risky_reward['safe_reward'])
        self.assertLess(safe_reward['safe_gap'], risky_reward['safe_gap'])

    def test_prediction_improves_with_updates(self):
        metric = 'system.cpu_percent'
        predictor = MultiPredictor(metrics=[metric], model_type='exponential')

        errors = []
        for _ in range(10):
            pred = predictor.predict_all()[metric]
            actual = 0.5
            error = abs(actual - pred)
            errors.append(error)
            predictor.update_all({metric: actual})

        self.assertLess(errors[-1], errors[0])

    def test_error_evaluator_accuracy(self):
        predictions = {'a': 10.0, 'b': 20.0}
        actuals = {'a': 10.0, 'b': 20.0}
        result = self.error_evaluator.evaluate(predictions, actuals)
        self.assertEqual(result['mae'], 0.0)
        self.assertEqual(result['mse'], 0.0)
        self.assertEqual(result['max_error'], 0.0)

    def test_optimizer_adjusts_params(self):
        predictor = MultiPredictor(metrics=['test'], model_type='exponential')
        predictor.update_all({'test': 10.0})

        old_params = predictor.get_metric_params('test')
        self.optimizer.optimize(predictor, {'test': 0.5})
        self.optimizer.optimize(predictor, {'test': 0.6})

        new_params = predictor.get_metric_params('test')
        self.assertNotEqual(old_params['alpha'], new_params['alpha'])


if __name__ == '__main__':
    unittest.main()
