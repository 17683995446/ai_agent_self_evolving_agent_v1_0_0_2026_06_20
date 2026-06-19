import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.planning.long_task_reviewer import LongTaskReviewer
from src.planning.multi_agent_runner import MultiAgentLongTaskRunner
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.memory.long_term import LongTermMemory
from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_learner import SkillLearner
from src.task.task_definition import Task


class TestLongTaskReviewer(unittest.TestCase):
    """LongTaskReviewer 单元测试"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = LongTermMemory(base_dir=os.path.join(self.tmp.name, 'ltm'))
        self.registry = SkillRegistry()
        self.learner = SkillLearner(self.registry)
        self.reviewer = LongTaskReviewer(self.memory, self.registry, self.learner)

    def tearDown(self):
        self.tmp.cleanup()

    def test_persist_mission(self):
        summary = {'plan_id': 'plan_001', 'status': 'completed', 'results': []}
        review_id = self.reviewer.persist_mission(summary)
        loaded = self.memory.load_knowledge(review_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['summary']['plan_id'], 'plan_001')

    def test_review_success_only(self):
        summary = {
            'plan_id': 'plan_ok',
            'status': 'completed',
            'results': [
                {'task_name': 'step_a', 'task_id': 't1', 'success': True, 'result': 'ok', 'duration': 0.01},
                {'task_name': 'step_b', 'task_id': 't2', 'success': True, 'result': 'ok', 'duration': 0.02},
            ],
        }
        report = self.reviewer.review(summary)
        self.assertEqual(report['total_steps'], 2)
        self.assertEqual(report['success_steps'], 2)
        self.assertEqual(report['failure_steps'], 0)
        self.assertEqual(report['success_rate'], 1.0)
        self.assertEqual(len(report['failure_attribution']), 0)
        # 复盘报告应持久化
        loaded = self.reviewer.load_review(report['review_id'])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['review_id'], report['review_id'])

    def test_review_with_failures(self):
        self.registry.register(Skill(name='flaky', action=lambda t: (_ for _ in ()).throw(RuntimeError('boom'))))
        summary = {
            'plan_id': 'plan_fail',
            'status': 'failed',
            'results': [
                {'task_name': 'ok_step', 'task_id': 't1', 'success': True, 'result': 'ok', 'duration': 0.01},
                {'task_name': 'flaky', 'task_id': 't2', 'success': False, 'result': 'boom', 'duration': 0.01, 'worker_id': 'w1'},
            ],
        }
        report = self.reviewer.review(summary)
        self.assertEqual(report['failure_steps'], 1)
        self.assertEqual(len(report['failure_attribution']), 1)
        self.assertEqual(report['failure_attribution'][0]['task_name'], 'flaky')
        self.assertEqual(report['failure_attribution'][0]['error'], 'boom')
        # 技能统计应通过 learner 更新
        skill = self.registry.get('flaky')
        self.assertEqual(skill.failure_count, 1)

    def test_review_history(self):
        for i in range(3):
            summary = {'plan_id': f'plan_{i}', 'status': 'completed', 'results': []}
            self.reviewer.review(summary)
        history = self.reviewer.review_history()
        self.assertEqual(len(history), 3)

    def test_generate_report(self):
        report = {
            'review_id': 'rev_001',
            'plan_id': 'plan_001',
            'status': 'completed',
            'total_steps': 3,
            'success_steps': 2,
            'failure_steps': 1,
            'success_rate': 2 / 3,
            'duration': 0.123,
            'failure_attribution': [{'task_name': 'bad', 'worker_id': 'w1', 'error': 'timeout'}],
            'skill_recommendations': {'deprecate': ['bad'], 'preferred': [], 'practice': []},
        }
        text = self.reviewer.generate_report(report)
        self.assertIn('Long Task Review Report', text)
        self.assertIn('bad', text)
        self.assertIn('timeout', text)


class TestEndToEndWithReview(unittest.TestCase):
    """端到端：多 Agent 长任务执行 + 自动复盘"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = LongTermMemory(base_dir=os.path.join(self.tmp.name, 'ltm'))
        self.orchestrator = MultiAgentOrchestrator(strategy='capability_match')
        self.orchestrator.register_worker(AgentWorker(worker_id='cpu_w', capabilities=['cpu']))
        self.orchestrator.register_worker(AgentWorker(worker_id='disk_w', capabilities=['disk']))
        self.registry = SkillRegistry()
        self.reviewer = LongTaskReviewer(self.memory, self.registry)
        self.runner = MultiAgentLongTaskRunner(
            self.orchestrator,
            registry=self.registry,
            memory=self.memory,
            reviewer=self.reviewer,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_pipeline_with_auto_review(self):
        self.runner.register_skill('compute', lambda t: t.params['x'] * 2, ['cpu'])
        self.runner.register_skill('save', lambda t: f"saved {t.params['x']}", ['disk'])

        goal = Task(name='pipeline')
        steps = [
            Task(name='compute', params={'required_capabilities': ['cpu'], 'x': 5}),
            Task(name='save', params={'required_capabilities': ['disk'], 'x': 5}),
        ]
        summary = self.runner.run(goal, steps, auto_review=True)

        self.assertEqual(summary['status'], 'completed')
        self.assertIn('review', summary)
        review = summary['review']
        self.assertEqual(review['success_steps'], 2)
        self.assertEqual(review['failure_steps'], 0)

        # 复盘报告应持久化到长期记忆
        loaded = self.reviewer.load_review(review['review_id'])
        self.assertIsNotNone(loaded)

        # runner.last_review 应返回最近复盘
        self.assertEqual(self.runner.last_review()['review_id'], review['review_id'])

    def test_review_drives_skill_feedback(self):
        self.runner.register_skill('flaky', lambda t: (_ for _ in ()).throw(ValueError('fail')), ['cpu'])
        goal = Task(name='fragile')
        steps = [Task(name='flaky', params={'required_capabilities': ['cpu']})]
        summary = self.runner.run(goal, steps, continue_on_failure=True, auto_review=True)

        review = summary['review']
        self.assertEqual(review['failure_steps'], 1)
        # 技能应通过复盘吸收失败反馈
        skill = self.registry.get('flaky')
        self.assertGreaterEqual(skill.failure_count, 1)

    def test_auto_review_disabled(self):
        self.runner.register_skill('compute', lambda t: 'ok', ['cpu'])
        goal = Task(name='simple')
        steps = [Task(name='compute', params={'required_capabilities': ['cpu']})]
        summary = self.runner.run(goal, steps, auto_review=False)
        self.assertNotIn('review', summary)
        self.assertIsNone(self.runner.last_review())


if __name__ == '__main__':
    unittest.main()
