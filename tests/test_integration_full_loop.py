import os
import sys
import unittest
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.state_vector import StateVector
from src.drive.reward_engine import RewardEngine
from src.task.task_pool import TaskPool
from src.prediction.multi_predictor import MultiPredictor
from src.prediction.error_evaluator import ErrorEvaluator
from src.optimization.micro_optimizer import MicroOptimizer
from src.memory.short_term import ShortTermMemory
from src.memory.review_engine import ReviewEngine


class TestFullAutonomousLoop(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_memory.db')

        self.state_vector = StateVector()
        self.reward_engine = RewardEngine()
        self.task_pool = TaskPool()
        self.predictor = MultiPredictor(metrics=['system.cpu_percent', 'system.mem_percent'])
        self.error_evaluator = ErrorEvaluator()
        self.optimizer = MicroOptimizer()
        self.memory = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        self.review_engine = ReviewEngine(self.memory, self.optimizer, min_samples=5)

    def tearDown(self):
        self.memory.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_loop_with_tasks(self):
        state = self.state_vector.perceive()
        reward_info = self.reward_engine.compute_total_reward(state, [])

        new_tasks = self.task_pool.generator.generate(reward_info)
        self.task_pool.update(new_tasks)
        self.assertGreater(len(self.task_pool.tasks), 0)

        self.task_pool.score_all(state, reward_info)
        selected_task = self.task_pool.select_task()
        self.assertIsNotNone(selected_task)

        predictions = self.predictor.predict_all()
        time.sleep(0.2)
        next_state = self.state_vector.perceive()
        actuals = {
            'system.cpu_percent': next_state['system']['cpu_percent'],
            'system.mem_percent': next_state['system']['mem_percent'],
        }

        error_result = self.error_evaluator.evaluate(predictions, actuals)
        self.predictor.update_all(actuals)
        self.optimizer.optimize(self.predictor, error_result['errors'])

        self.task_pool.mark_executed(selected_task.id, success=True, reward=reward_info['total_reward'])

        sample = {
            'timestamp': time.time(),
            'state': state,
            'predictions': predictions,
            'actuals': actuals,
            'errors': error_result['errors'],
            'mae': error_result['mae'],
            'task_id': selected_task.id,
            'reward': reward_info['total_reward'],
        }
        self.memory.store_sample(sample)
        self.assertEqual(self.memory.get_sample_count(), 1)

    def test_multiple_iterations(self):
        for i in range(5):
            state = self.state_vector.perceive()
            reward_info = self.reward_engine.compute_total_reward(state, [])

            new_tasks = self.task_pool.generator.generate(reward_info)
            self.task_pool.update(new_tasks)
            self.task_pool.score_all(state, reward_info)
            selected_task = self.task_pool.select_task()

            predictions = self.predictor.predict_all()
            time.sleep(0.1)
            next_state = self.state_vector.perceive()
            actuals = {
                'system.cpu_percent': next_state['system']['cpu_percent'],
                'system.mem_percent': next_state['system']['mem_percent'],
            }

            error_result = self.error_evaluator.evaluate(predictions, actuals)
            self.predictor.update_all(actuals)
            self.optimizer.optimize(self.predictor, error_result['errors'])

            if selected_task:
                self.task_pool.mark_executed(selected_task.id, success=True, reward=reward_info['total_reward'])

                sample = {
                    'timestamp': time.time(),
                    'state': state,
                    'predictions': predictions,
                    'actuals': actuals,
                    'errors': error_result['errors'],
                    'mae': error_result['mae'],
                    'task_id': selected_task.id,
                    'reward': reward_info['total_reward'],
                }
                self.memory.store_sample(sample)

        self.assertGreaterEqual(self.memory.get_sample_count(), 5)

    def test_gap_drives_task_generation(self):
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

        safe_tasks = self.task_pool.generator.generate(safe_reward)
        risky_tasks = self.task_pool.generator.generate(risky_reward)

        self.assertGreater(len(risky_tasks), len(safe_tasks))
        for task in risky_tasks:
            self.assertEqual(task.type.value, 'safe')

    def test_offline_review_trigger(self):
        for i in range(10):
            sample = {
                'timestamp': time.time(),
                'state': {},
                'predictions': {},
                'actuals': {},
                'errors': {'cpu': 0.1},
                'mae': 0.1,
                'task_id': 'task_001',
                'reward': 0.8,
            }
            self.memory.store_sample(sample)

        should_run = self.review_engine.should_review(self.memory.get_sample_count(), idle_cpu=15.0)
        self.assertTrue(should_run)

        result = self.review_engine.run_review()
        self.assertEqual(result['status'], 'completed')

    def test_task_pool_size_management(self):
        for i in range(20):
            tasks = self.task_pool.generator.generate({'safe_gap': 0.9, 'gain_gap': 0.9, 'eff_gap': 0.9})
            self.task_pool.update(tasks)

        self.assertLessEqual(len(self.task_pool.tasks), self.task_pool.max_size)

    def test_prediction_error_decreases(self):
        metric = 'system.cpu_percent'
        predictor = MultiPredictor(metrics=[metric])

        errors = []
        for _ in range(10):
            pred = predictor.predict_all()[metric]
            actual = 0.5
            error = abs(actual - pred)
            errors.append(error)
            predictor.update_all({metric: actual})

        self.assertLess(errors[-1], errors[0])

    def test_memory_cleanup(self):
        for i in range(15):
            sample = {
                'timestamp': time.time(),
                'state': {},
                'predictions': {},
                'actuals': {},
                'errors': {'cpu': i * 0.01},
                'mae': i * 0.01,
                'task_id': 'task_001',
                'reward': 0.5,
                'value_score': float(i) / 15,
            }
            self.memory.store_sample(sample)

        count_before = self.memory.get_sample_count()
        # Force cleanup by temporarily lowering max_samples
        original_max = self.memory.max_samples
        self.memory.max_samples = 5
        deleted = self.memory.cleanup_old_samples(retention_ratio=0.3)
        self.memory.max_samples = original_max
        count_after = self.memory.get_sample_count()

        self.assertGreater(deleted, 0)
        self.assertLess(count_after, count_before)


if __name__ == '__main__':
    unittest.main()
