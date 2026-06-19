import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.task.task_definition import Task
from src.core.constants import TaskType


class TestTaskDefinition(unittest.TestCase):
    def test_default_creation(self):
        task = Task()
        self.assertTrue(task.id.startswith('task_'))
        self.assertEqual(task.type, TaskType.SAFE)
        self.assertEqual(task.name, 'unnamed')
        self.assertEqual(task.score, 0.0)
        self.assertEqual(task.execution_count, 0)
        self.assertFalse(task.frozen)

    def test_custom_creation(self):
        task = Task(
            type=TaskType.GAIN,
            name='explore_cpu',
            description='Explore CPU patterns',
            params={'target': 'cpu'},
        )
        self.assertEqual(task.type, TaskType.GAIN)
        self.assertEqual(task.name, 'explore_cpu')
        self.assertEqual(task.params['target'], 'cpu')

    def test_success_rate_initial(self):
        task = Task()
        self.assertEqual(task.success_rate, 0.5)

    def test_success_rate_after_executions(self):
        task = Task()
        task.mark_executed(success=True)
        task.mark_executed(success=True)
        task.mark_executed(success=False)
        self.assertEqual(task.success_rate, 2 / 3)

    def test_mark_executed_updates_time(self):
        task = Task()
        self.assertIsNone(task.last_executed)
        task.mark_executed()
        self.assertIsNotNone(task.last_executed)
        self.assertGreater(task.last_executed, 0)

    def test_mark_executed_updates_reward(self):
        task = Task()
        task.mark_executed(success=True, reward=1.0)
        self.assertEqual(task.avg_reward, 1.0)
        task.mark_executed(success=True, reward=0.5)
        self.assertAlmostEqual(task.avg_reward, 0.85, places=2)

    def test_freeze_unfreeze(self):
        task = Task()
        self.assertFalse(task.frozen)
        task.freeze()
        self.assertTrue(task.frozen)
        task.unfreeze()
        self.assertFalse(task.frozen)

    def test_is_exploration(self):
        safe_task = Task(type=TaskType.SAFE)
        explore_task = Task(type=TaskType.EXPLORE)
        self.assertFalse(safe_task.is_exploration)
        self.assertTrue(explore_task.is_exploration)

    def test_to_dict(self):
        task = Task(type=TaskType.EFF, name='cleanup')
        data = task.to_dict()
        self.assertEqual(data['type'], 'eff')
        self.assertEqual(data['name'], 'cleanup')
        self.assertIn('id', data)
        self.assertIn('created_at', data)

    def test_from_dict(self):
        data = {
            'id': 'task_abc123',
            'type': 'gain',
            'name': 'test_task',
            'description': 'A test task',
            'params': {'key': 'value'},
            'score': 0.8,
            'execution_count': 5,
            'success_count': 4,
            'frozen': True,
            'avg_reward': 0.9,
        }
        task = Task.from_dict(data)
        self.assertEqual(task.id, 'task_abc123')
        self.assertEqual(task.type, TaskType.GAIN)
        self.assertEqual(task.name, 'test_task')
        self.assertEqual(task.score, 0.8)
        self.assertEqual(task.execution_count, 5)
        self.assertEqual(task.success_count, 4)
        self.assertTrue(task.frozen)
        self.assertEqual(task.avg_reward, 0.9)

    def test_roundtrip_serialization(self):
        original = Task(type=TaskType.SAFE, name='monitor')
        original.mark_executed(success=True, reward=0.8)
        data = original.to_dict()
        restored = Task.from_dict(data)
        self.assertEqual(restored.type, original.type)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.execution_count, original.execution_count)
        self.assertEqual(restored.success_count, original.success_count)


if __name__ == '__main__':
    unittest.main()
