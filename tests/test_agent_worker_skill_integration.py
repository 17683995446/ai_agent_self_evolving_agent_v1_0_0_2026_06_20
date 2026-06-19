import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.multi_agent.agent_worker import AgentWorker
from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.task.task_definition import Task


class TestAgentWorkerSkillIntegration(unittest.TestCase):
    """AgentWorker 与 Skill 深度集成测试"""

    def test_use_skill_without_registry_returns_none(self):
        worker = AgentWorker(worker_id='w1')
        task = Task(name='noop')
        self.assertIsNone(worker.use_skill(task))

    def test_attach_skill_registry_enables_use_skill(self):
        worker = AgentWorker(worker_id='w1', capabilities=['cpu'])
        registry = SkillRegistry()
        registry.register(Skill(name='compute', capabilities=['cpu'], action=lambda t: t.params['x'] * 2))
        worker.attach_skill_registry(registry)

        task = Task(name='compute', params={'required_capabilities': ['cpu'], 'x': 5})
        record = worker.use_skill(task)
        self.assertIsNotNone(record)
        self.assertTrue(record['success'])
        self.assertEqual(record['result'], 10)
        self.assertTrue(record['skill_used'])

    def test_use_skill_updates_skill_stats(self):
        worker = AgentWorker(worker_id='w1', capabilities=['cpu'])
        registry = SkillRegistry()
        skill = Skill(name='compute', capabilities=['cpu'], action=lambda t: t.params['x'] + 1)
        registry.register(skill)
        worker.attach_skill_registry(registry)

        for i in range(3):
            worker.use_skill(Task(name='compute', params={'required_capabilities': ['cpu'], 'x': i}))

        # 技能统计应由 learner 统一更新
        self.assertEqual(skill.success_count, 3)
        self.assertEqual(skill.failure_count, 0)
        self.assertEqual(skill.usage_count, 3)

    def test_use_skill_with_failing_action(self):
        worker = AgentWorker(worker_id='w1', capabilities=['cpu'])
        registry = SkillRegistry()
        skill = Skill(name='fail', capabilities=['cpu'], action=lambda t: 1 / 0)
        registry.register(skill)
        worker.attach_skill_registry(registry)

        record = worker.use_skill(Task(name='fail', params={'required_capabilities': ['cpu']}))
        self.assertIsNotNone(record)
        self.assertFalse(record['success'])
        self.assertEqual(skill.failure_count, 1)

    def test_use_skill_no_matching_skill(self):
        worker = AgentWorker(worker_id='w1', capabilities=['cpu'])
        registry = SkillRegistry()
        registry.register(Skill(name='disk_skill', capabilities=['disk'], action=lambda t: 'disk'))
        worker.attach_skill_registry(registry)

        # 任务要求 cpu 能力，但注册表中只有 disk 技能
        task = Task(name='cpu_task', params={'required_capabilities': ['cpu']})
        self.assertIsNone(worker.use_skill(task))

    def test_stats_reflect_skill_usage(self):
        worker = AgentWorker(worker_id='w1', capabilities=['cpu'])
        registry = SkillRegistry()
        registry.register(Skill(name='compute', capabilities=['cpu'], action=lambda t: 'ok'))
        worker.attach_skill_registry(registry)

        worker.use_skill(Task(name='compute', params={'required_capabilities': ['cpu']}))
        worker.execute_task(Task(name='plain'))

        stats = worker.get_stats()
        self.assertEqual(stats['total_executions'], 2)
        self.assertEqual(stats['skill_used_count'], 1)
        self.assertTrue(stats['skill_registry_attached'])

    def test_registry_provided_in_constructor(self):
        registry = SkillRegistry()
        registry.register(Skill(name='gen', action=lambda t: 'gen'))
        worker = AgentWorker(worker_id='w1', skill_registry=registry)

        record = worker.use_skill(Task(name='gen'))
        self.assertIsNotNone(record)
        self.assertTrue(record['success'])
        self.assertEqual(record['result'], 'gen')

    def test_use_skill_marks_task_executed(self):
        worker = AgentWorker(worker_id='w1', capabilities=['cpu'])
        registry = SkillRegistry()
        registry.register(Skill(name='compute', capabilities=['cpu'], action=lambda t: t.params['x']))
        worker.attach_skill_registry(registry)

        task = Task(name='compute', params={'required_capabilities': ['cpu'], 'x': 42})
        worker.use_skill(task)
        self.assertEqual(task.execution_count, 1)
        self.assertEqual(task.success_count, 1)


if __name__ == '__main__':
    unittest.main()
