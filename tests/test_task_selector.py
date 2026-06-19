import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.task.task_selector import TaskSelector
from src.task.task_definition import Task
from src.core.constants import TaskType


class TestTaskSelector(unittest.TestCase):
    def setUp(self):
        self.selector = TaskSelector(explore_ratio=0.3)

    def test_select_empty_pool(self):
        result = self.selector.select([])
        self.assertIsNone(result)

    def test_select_all_frozen(self):
        tasks = [Task(type=TaskType.SAFE) for _ in range(3)]
        for t in tasks:
            t.freeze()
        result = self.selector.select(tasks)
        self.assertIsNone(result)

    def test_exploit_selects_highest_score(self):
        self.selector.current_ratio = 0.0
        tasks = [
            Task(type=TaskType.SAFE, name='low'),
            Task(type=TaskType.SAFE, name='high'),
            Task(type=TaskType.SAFE, name='medium'),
        ]
        tasks[0].score = 0.2
        tasks[1].score = 0.9
        tasks[2].score = 0.5
        result = self.selector.select(tasks)
        self.assertEqual(result.name, 'high')

    def test_explore_selects_low_score(self):
        self.selector.current_ratio = 1.0
        tasks = [
            Task(type=TaskType.SAFE, name='low'),
            Task(type=TaskType.SAFE, name='high'),
        ]
        tasks[0].score = 0.1
        tasks[1].score = 0.9
        result = self.selector.select(tasks)
        self.assertEqual(result.name, 'low')

    def test_adjust_ratio_emergency(self):
        ratio = self.selector.adjust_ratio(0.1, is_emergency=True)
        self.assertEqual(ratio, 0.1)

    def test_adjust_ratio_novel(self):
        ratio = self.selector.adjust_ratio(0.1, is_novel_environment=True)
        self.assertEqual(ratio, 0.4)

    def test_adjust_ratio_stable(self):
        ratio = self.selector.adjust_ratio(0.01)
        self.assertEqual(ratio, 0.5)

    def test_adjust_ratio_normal(self):
        ratio = self.selector.adjust_ratio(0.2)
        self.assertEqual(ratio, 0.3)

    def test_get_ratio(self):
        self.assertEqual(self.selector.get_ratio(), 0.3)
        self.selector.adjust_ratio(0.01)
        self.assertEqual(self.selector.get_ratio(), 0.5)

    def test_select_returns_task(self):
        tasks = [Task(type=TaskType.SAFE) for _ in range(3)]
        for t in tasks:
            t.score = 0.5
        result = self.selector.select(tasks)
        self.assertIsInstance(result, Task)
        self.assertIn(result, tasks)


if __name__ == '__main__':
    unittest.main()
