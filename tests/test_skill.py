import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_learner import SkillLearner
from src.task.task_definition import Task


class TestSkill(unittest.TestCase):
    """Skill 基础单元测试"""

    def test_skill_execute_with_action(self):
        skill = Skill(name='add', action=lambda t: t.params['x'] + t.params['y'])
        task = Task(name='add', params={'x': 1, 'y': 2})
        result = skill.execute(task)
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 3)
        self.assertEqual(skill.success_rate, 1.0)

    def test_skill_execute_failure(self):
        skill = Skill(name='fail', action=lambda t: 1 / 0)
        task = Task(name='fail')
        result = skill.execute(task)
        self.assertFalse(result['success'])
        self.assertEqual(skill.success_rate, 0.0)

    def test_skill_can_handle(self):
        skill = Skill(name='disk_clean', capabilities=['disk'])
        task = Task(name='clean', params={'required_capabilities': ['disk']})
        self.assertTrue(skill.can_handle(task))
        task2 = Task(name='clean', params={'required_capabilities': ['network']})
        self.assertFalse(skill.can_handle(task2))

    def test_skill_placeholder(self):
        skill = Skill(name='noop')
        result = skill.execute(Task(name='noop'))
        self.assertTrue(result['success'])
        self.assertIn('placeholder', result['result'])


class TestSkillRegistry(unittest.TestCase):
    """SkillRegistry 单元测试"""

    def setUp(self):
        self.registry = SkillRegistry()
        self.s1 = Skill(name='cpu_skill', capabilities=['cpu'], action=lambda t: 'cpu')
        self.s2 = Skill(name='disk_skill', capabilities=['disk'], action=lambda t: 'disk')
        self.registry.register(self.s1)
        self.registry.register(self.s2)

    def test_register_and_get(self):
        self.assertEqual(self.registry.get('cpu_skill'), self.s1)
        self.assertIsNone(self.registry.get('missing'))
        self.assertFalse(self.registry.register(self.s1))

    def test_unregister_and_list(self):
        self.assertTrue(self.registry.unregister('cpu_skill'))
        self.assertEqual(sorted(self.registry.list_skills()), ['disk_skill'])

    def test_find_by_capability(self):
        found = self.registry.find_by_capability('disk')
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, 'disk_skill')

    def test_select_best_by_success_rate(self):
        s3 = Skill(name='expert', action=lambda t: 'expert')
        # expert 无能力要求，可处理任何任务
        for _ in range(5):
            s3.execute(Task(name='expert'))
        self.registry.register(s3)
        task = Task(name='generic')
        best = self.registry.select_best(task)
        self.assertEqual(best.name, 'expert')

    def test_select_best_with_capability(self):
        task = Task(name='cleanup', params={'required_capabilities': ['disk']})
        best = self.registry.select_best(task)
        self.assertEqual(best.name, 'disk_skill')

    def test_execute(self):
        task = Task(name='cleanup', params={'required_capabilities': ['disk']})
        result = self.registry.execute(task)
        self.assertIsNotNone(result)
        self.assertEqual(result['result'], 'disk')

    def test_update_skill(self):
        self.assertTrue(self.registry.update_skill('cpu_skill', description='updated'))
        self.assertEqual(self.registry.get('cpu_skill').description, 'updated')
        self.assertFalse(self.registry.update_skill('missing'))

    def test_to_dict(self):
        self.s1.execute(Task(name='t'))
        data = self.registry.to_dict()
        self.assertEqual(data['skill_count'], 2)
        self.assertEqual(data['total_success'], 1)


class TestSkillLearner(unittest.TestCase):
    """SkillLearner 自学习单元测试"""

    def test_auto_register(self):
        registry = SkillRegistry()
        learner = SkillLearner(registry)
        task = Task(name='double', params={'value': 3})
        skill = learner.auto_register(task, action=lambda t: t.params['value'] * 2)
        self.assertEqual(skill.name, 'double')
        self.assertIn('double', registry.list_skills())

    def test_feedback_and_improve(self):
        registry = SkillRegistry()
        learner = SkillLearner(registry)
        skill = Skill(name='flaky', action=lambda t: (_ for _ in ()).throw(RuntimeError('boom')))
        registry.register(skill)
        for _ in range(5):
            skill.execute(Task(name='flaky'))
        rec = learner.improve()
        self.assertIn('flaky', rec['deprecate'])

    def test_learn_from_record(self):
        registry = SkillRegistry()
        learner = SkillLearner(registry)
        registry.register(Skill(name='rec', action=lambda t: 'ok'))
        self.assertTrue(learner.learn_from_record({'task_name': 'rec', 'success': True}))
        self.assertEqual(registry.get('rec').success_rate, 1.0)

    def test_suggest(self):
        registry = SkillRegistry()
        learner = SkillLearner(registry)
        registry.register(Skill(name='match', capabilities=['cpu'], action=lambda t: 'cpu'))
        task = Task(name='job', params={'required_capabilities': ['cpu']})
        self.assertEqual(learner.suggest(task).name, 'match')


if __name__ == '__main__':
    unittest.main()
