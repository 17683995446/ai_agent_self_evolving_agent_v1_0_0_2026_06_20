import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.task.task_definition import Task
from src.core.constants import TaskType


class TestAgentWorker(unittest.TestCase):
    """AgentWorker 单元测试"""

    def test_default_worker_id(self):
        w = AgentWorker()
        self.assertTrue(w.worker_id.startswith('worker_'))

    def test_capabilities(self):
        w = AgentWorker(capabilities=['cpu', 'mem'])
        self.assertTrue(w.has_capability('cpu'))
        self.assertFalse(w.has_capability('disk'))
        w.add_capability('disk')
        self.assertTrue(w.has_capability('disk'))

    def test_can_handle_without_requirements(self):
        w = AgentWorker(capabilities=['cpu'])
        task = Task(name='monitor', type=TaskType.SAFE)
        self.assertTrue(w.can_handle(task))

    def test_can_handle_with_requirements(self):
        w = AgentWorker(capabilities=['cpu', 'mem'])
        task = Task(name='monitor', type=TaskType.SAFE, params={'required_capabilities': ['cpu']})
        self.assertTrue(w.can_handle(task))
        task2 = Task(name='monitor', type=TaskType.SAFE, params={'required_capabilities': ['disk']})
        self.assertFalse(w.can_handle(task2))

    def test_execute_task_with_action(self):
        w = AgentWorker()
        task = Task(name='add', type=TaskType.GAIN, params={'x': 1, 'y': 2})
        result = w.execute_task(task, action=lambda t: t.params['x'] + t.params['y'])
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 3)
        self.assertEqual(result['worker_id'], w.worker_id)

    def test_execute_task_failure(self):
        w = AgentWorker()
        task = Task(name='fail', type=TaskType.SAFE)
        result = w.execute_task(task, action=lambda t: 1 / 0)
        self.assertFalse(result['success'])
        self.assertEqual(w.get_stats()['success_rate'], 0.0)

    def test_stats(self):
        w = AgentWorker()
        stats = w.get_stats()
        self.assertEqual(stats['total_executions'], 0)
        self.assertEqual(stats['success_rate'], 0.0)


class TestMultiAgentOrchestrator(unittest.TestCase):
    """MultiAgentOrchestrator 单元测试"""

    def setUp(self):
        self.orchestrator = MultiAgentOrchestrator()
        self.w1 = AgentWorker(worker_id='w1', capabilities=['cpu', 'mem'])
        self.w2 = AgentWorker(worker_id='w2', capabilities=['disk', 'network'])
        self.orchestrator.register_worker(self.w1)
        self.orchestrator.register_worker(self.w2)

    def test_register_and_list_workers(self):
        self.assertEqual(sorted(self.orchestrator.list_workers()), ['w1', 'w2'])
        self.assertTrue(self.orchestrator.unregister_worker('w1'))
        self.assertEqual(self.orchestrator.list_workers(), ['w2'])

    def test_round_robin_assignment(self):
        t1 = Task(name='t1')
        t2 = Task(name='t2')
        r1 = self.orchestrator.assign_task(t1)
        r2 = self.orchestrator.assign_task(t2)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        self.assertNotEqual(r1['worker_id'], r2['worker_id'])

    def test_capability_match_no_candidate(self):
        orch = MultiAgentOrchestrator(strategy='capability_match')
        orch.register_worker(self.w1)
        task = Task(name='disk_task', params={'required_capabilities': ['disk']})
        result = orch.assign_task(task)
        self.assertIsNone(result)

    def test_capability_match_selects_correct_worker(self):
        orch = MultiAgentOrchestrator(strategy='capability_match')
        orch.register_worker(self.w1)
        orch.register_worker(self.w2)
        task = Task(name='disk_task', params={'required_capabilities': ['disk']})
        result = orch.assign_task(task)
        self.assertIsNotNone(result)
        self.assertEqual(result['worker_id'], 'w2')

    def test_least_loaded_strategy(self):
        orch = MultiAgentOrchestrator(strategy='least_loaded')
        orch.register_worker(self.w1)
        orch.register_worker(self.w2)
        # 预先让 w1 多一个任务
        self.w1.task_pool.update([Task(name='preload')])
        task = Task(name='load_test')
        result = orch.assign_task(task)
        self.assertEqual(result['worker_id'], 'w2')

    def test_run_mission_aggregates_results(self):
        tasks = [Task(name=f't{i}') for i in range(4)]
        summary = self.orchestrator.run_mission(tasks)
        self.assertEqual(summary['total_tasks'], 4)
        self.assertEqual(summary['completed_tasks'], 4)
        self.assertEqual(summary['failed_tasks'], 0)
        self.assertIn('mission_id', summary)
        self.assertEqual(len(summary['results']), 4)

    def test_run_mission_with_actions(self):
        tasks = [Task(name='double', params={'value': i}) for i in range(3)]
        actions = {'double': lambda t: t.params['value'] * 2}
        summary = self.orchestrator.run_mission(tasks, actions=actions)
        results = [r['result'] for r in summary['results']]
        self.assertIn(0, results)
        self.assertIn(2, results)
        self.assertIn(4, results)

    def test_mission_history(self):
        tasks = [Task(name='h1')]
        summary = self.orchestrator.run_mission(tasks, mission_id='m_001')
        self.assertEqual(self.orchestrator.list_missions(), ['m_001'])
        self.assertEqual(self.orchestrator.get_mission('m_001'), summary)

    def test_aggregated_stats(self):
        tasks = [Task(name='s1'), Task(name='s2')]
        self.orchestrator.run_mission(tasks)
        stats = self.orchestrator.get_aggregated_stats()
        self.assertEqual(stats['worker_count'], 2)
        self.assertEqual(stats['total_executions'], 2)
        self.assertEqual(stats['overall_success_rate'], 1.0)

    def test_unsupported_strategy(self):
        with self.assertRaises(ValueError):
            MultiAgentOrchestrator(strategy='unknown')


if __name__ == '__main__':
    unittest.main()
