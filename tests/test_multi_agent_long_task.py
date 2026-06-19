import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.planning.multi_agent_runner import MultiAgentLongTaskRunner
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.task.task_definition import Task


class TestMultiAgentLongTask(unittest.TestCase):
    """多 Agent 长任务集成测试"""

    def setUp(self):
        self.orchestrator = MultiAgentOrchestrator(strategy='capability_match')
        self.orchestrator.register_worker(AgentWorker(worker_id='net_w', capabilities=['network']))
        self.orchestrator.register_worker(AgentWorker(worker_id='cpu_w', capabilities=['cpu']))
        self.orchestrator.register_worker(AgentWorker(worker_id='disk_w', capabilities=['disk']))
        self.runner = MultiAgentLongTaskRunner(self.orchestrator)

    def test_pipeline_dispatched_to_matching_workers(self):
        self.runner.register_skill('collect', lambda t: f"collected from {t.params.get('source')}", ['network'])
        self.runner.register_skill('transform', lambda t: f"transformed with {t.params.get('algo')}", ['cpu'])
        self.runner.register_skill('store', lambda t: f"stored to {t.params.get('path')}", ['disk'])

        goal = Task(name='data_pipeline')
        steps = [
            Task(name='collect', params={'required_capabilities': ['network'], 'source': 'api'}),
            Task(name='transform', params={'required_capabilities': ['cpu'], 'algo': 'normalize'}),
            Task(name='store', params={'required_capabilities': ['disk'], 'path': '/data'}),
        ]
        summary = self.runner.run(goal, steps)

        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['total_steps'], 3)
        self.assertEqual(summary['completed_steps'], 3)

        worker_ids = {r['worker_id'] for r in summary['results']}
        self.assertEqual(worker_ids, {'net_w', 'cpu_w', 'disk_w'})

        stats = self.runner.skill_stats()
        self.assertEqual(stats['skill_count'], 3)
        self.assertEqual(stats['total_success'], 3)

    def test_failure_is_propagated_and_learned(self):
        self.runner.register_skill('bad_skill', lambda t: (_ for _ in ()).throw(RuntimeError('boom')), ['cpu'])
        goal = Task(name='fragile_pipeline')
        steps = [Task(name='bad_skill', params={'required_capabilities': ['cpu']})]
        summary = self.runner.run(goal, steps, continue_on_failure=True)

        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['failed_steps'], 1)
        stats = self.runner.skill_stats()
        self.assertEqual(stats['total_failure'], 1)

    def test_runner_uses_default_skill_when_no_explicit_action(self):
        self.runner.register_skill('generic', lambda t: 'done', [])
        goal = Task(name='simple')
        steps = [Task(name='generic')]
        summary = self.runner.run(goal, steps)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['results'][0]['result'], 'done')


if __name__ == '__main__':
    unittest.main()
