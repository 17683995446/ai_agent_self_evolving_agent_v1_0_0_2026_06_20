import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.task.task_pool import TaskPool
from src.task.task_definition import Task
from src.core.constants import TaskType


class TestTaskPool(unittest.TestCase):
    def setUp(self):
        self.pool = TaskPool(max_size=10, max_age_hours=24.0)

    def test_add_tasks(self):
        tasks = [Task(type=TaskType.SAFE) for _ in range(3)]
        self.pool.update(tasks)
        self.assertEqual(len(self.pool.tasks), 3)

    def test_no_duplicate_tasks(self):
        task = Task(type=TaskType.SAFE)
        self.pool.update([task])
        self.pool.update([task])
        self.assertEqual(len(self.pool.tasks), 1)

    def test_select_task(self):
        tasks = [Task(type=TaskType.SAFE) for _ in range(3)]
        for t in tasks:
            t.score = 0.5
        self.pool.update(tasks)
        result = self.pool.select_task()
        self.assertIsNotNone(result)
        self.assertIn(result, tasks)

    def test_mark_executed(self):
        task = Task(type=TaskType.SAFE)
        self.pool.update([task])
        self.pool.mark_executed(task.id, success=True, reward=1.0)
        self.assertEqual(self.pool.tasks[task.id].execution_count, 1)
        self.assertEqual(self.pool.tasks[task.id].success_count, 1)

    def test_freeze_unfreeze(self):
        task = Task(type=TaskType.SAFE)
        self.pool.update([task])
        self.pool.freeze_task(task.id)
        self.assertTrue(self.pool.tasks[task.id].frozen)
        self.pool.unfreeze_task(task.id)
        self.assertFalse(self.pool.tasks[task.id].frozen)

    def test_remove_task(self):
        task = Task(type=TaskType.SAFE)
        self.pool.update([task])
        result = self.pool.remove_task(task.id)
        self.assertTrue(result)
        self.assertEqual(len(self.pool.tasks), 0)

    def test_remove_nonexistent(self):
        result = self.pool.remove_task('nonexistent')
        self.assertFalse(result)

    def test_get_tasks_by_type(self):
        safe_task = Task(type=TaskType.SAFE)
        gain_task = Task(type=TaskType.GAIN)
        self.pool.update([safe_task, gain_task])
        safe_tasks = self.pool.get_tasks_by_type(TaskType.SAFE)
        self.assertEqual(len(safe_tasks), 1)
        self.assertEqual(safe_tasks[0].type, TaskType.SAFE)

    def test_get_frozen_tasks(self):
        task = Task(type=TaskType.SAFE)
        self.pool.update([task])
        self.pool.freeze_task(task.id)
        frozen = self.pool.get_frozen_tasks()
        self.assertEqual(len(frozen), 1)

    def test_get_top_tasks(self):
        tasks = [
            Task(type=TaskType.SAFE, name='low'),
            Task(type=TaskType.SAFE, name='high'),
        ]
        tasks[0].score = 0.2
        tasks[1].score = 0.9
        self.pool.update(tasks)
        top = self.pool.get_top_tasks(n=1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].name, 'high')

    def test_size_limit(self):
        tasks = [Task(type=TaskType.SAFE) for _ in range(15)]
        for i, t in enumerate(tasks):
            t.score = float(i) / 15
        self.pool.update(tasks)
        self.assertLessEqual(len(self.pool.tasks), 10)

    def test_stale_removal(self):
        task = Task(type=TaskType.SAFE)
        task.created_at = time.time() - 48 * 3600  # 48 hours ago
        self.pool.update([task])
        self.pool._remove_stale_tasks()
        self.assertEqual(len(self.pool.tasks), 0)

    def test_stale_not_removed_if_executed(self):
        task = Task(type=TaskType.SAFE)
        task.created_at = time.time() - 48 * 3600
        task.execution_count = 1
        self.pool.update([task])
        self.pool._remove_stale_tasks()
        self.assertEqual(len(self.pool.tasks), 1)

    def test_get_stats(self):
        tasks = [Task(type=TaskType.SAFE) for _ in range(3)]
        self.pool.update(tasks)
        stats = self.pool.get_stats()
        self.assertEqual(stats['total_tasks'], 3)
        self.assertEqual(stats['frozen_tasks'], 0)
        self.assertEqual(stats['active_tasks'], 3)
        self.assertIn('by_type', stats)

    def test_execution_history(self):
        task = Task(type=TaskType.SAFE)
        self.pool.update([task])
        self.pool.mark_executed(task.id, success=True, reward=1.0)
        self.pool.mark_executed(task.id, success=False, reward=0.0)
        stats = self.pool.get_stats()
        self.assertEqual(stats['total_executions'], 2)


if __name__ == '__main__':
    unittest.main()
