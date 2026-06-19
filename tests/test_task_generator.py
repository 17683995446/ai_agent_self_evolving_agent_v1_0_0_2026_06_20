import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.task.task_generator import TaskGenerator
from src.core.constants import TaskType


class TestTaskGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = TaskGenerator()

    def test_no_gaps_no_tasks(self):
        gaps = {'safe_gap': 0.0, 'gain_gap': 0.0, 'eff_gap': 0.0}
        tasks = self.generator.generate(gaps)
        self.assertEqual(len(tasks), 0)

    def test_safe_gap_generates_safe_tasks(self):
        gaps = {'safe_gap': 0.5, 'gain_gap': 0.0, 'eff_gap': 0.0}
        tasks = self.generator.generate(gaps)
        self.assertGreater(len(tasks), 0)
        for task in tasks:
            self.assertEqual(task.type, TaskType.SAFE)

    def test_gain_gap_generates_gain_tasks(self):
        gaps = {'safe_gap': 0.0, 'gain_gap': 0.5, 'eff_gap': 0.0}
        tasks = self.generator.generate(gaps)
        self.assertGreater(len(tasks), 0)
        for task in tasks:
            self.assertEqual(task.type, TaskType.GAIN)

    def test_eff_gap_generates_eff_tasks(self):
        gaps = {'safe_gap': 0.0, 'gain_gap': 0.0, 'eff_gap': 0.5}
        tasks = self.generator.generate(gaps)
        self.assertGreater(len(tasks), 0)
        for task in tasks:
            self.assertEqual(task.type, TaskType.EFF)

    def test_multiple_gaps_generate_mixed_tasks(self):
        gaps = {'safe_gap': 0.5, 'gain_gap': 0.5, 'eff_gap': 0.5}
        tasks = self.generator.generate(gaps)
        self.assertGreaterEqual(len(tasks), 3)
        types = {task.type for task in tasks}
        self.assertIn(TaskType.SAFE, types)
        self.assertIn(TaskType.GAIN, types)
        self.assertIn(TaskType.EFF, types)

    def test_large_gap_generates_more_tasks(self):
        small_gap = {'safe_gap': 0.35, 'gain_gap': 0.0, 'eff_gap': 0.0}
        large_gap = {'safe_gap': 0.8, 'gain_gap': 0.0, 'eff_gap': 0.0}
        small_tasks = self.generator.generate(small_gap)
        large_tasks = self.generator.generate(large_gap)
        self.assertLessEqual(len(small_tasks), len(large_tasks))

    def test_max_tasks_per_type(self):
        gaps = {'safe_gap': 1.0, 'gain_gap': 0.0, 'eff_gap': 0.0}
        tasks = self.generator.generate(gaps)
        self.assertLessEqual(len(tasks), 3)

    def test_task_has_action(self):
        gaps = {'safe_gap': 0.5, 'gain_gap': 0.0, 'eff_gap': 0.0}
        tasks = self.generator.generate(gaps)
        for task in tasks:
            self.assertIsNotNone(task.action)
            self.assertTrue(callable(task.action))

    def test_task_has_params(self):
        gaps = {'safe_gap': 0.5, 'gain_gap': 0.0, 'eff_gap': 0.0}
        tasks = self.generator.generate(gaps)
        for task in tasks:
            self.assertIsInstance(task.params, dict)
            self.assertGreater(len(task.params), 0)

    def test_add_custom_template(self):
        def custom_action():
            return True
        self.generator.add_safe_template('aaa_custom', custom_action)
        # Verify the template was added
        self.assertIn('aaa_custom', self.generator._safe_actions)
        # Verify we can generate tasks from it by requesting more than default count
        tasks = self.generator._generate_safe_tasks(7)
        names = [t.name for t in tasks]
        self.assertIn('aaa_custom', names)

    def test_task_execution(self):
        gaps = {'safe_gap': 0.5, 'gain_gap': 0.0, 'eff_gap': 0.0}
        tasks = self.generator.generate(gaps)
        for task in tasks:
            result = task.action()
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
