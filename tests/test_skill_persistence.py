"""阶段148：技能持久化与跨任务迁移测试。"""

import os
import shutil
import tempfile
import unittest

from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_persistence import SkillPersistence
from src.memory.long_term import LongTermMemory
from src.task.task_definition import Task


# 命名函数，可被 pickle 序列化
def named_fetch_action(task):
    return f"fetched {task.params.get('source', '')}"


def named_normalize_action(task):
    return f"normalized {task.params.get('records', 0)}"


class TestSkillPersistence(unittest.TestCase):
    """SkillPersistence 持久化与加载测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.memory = LongTermMemory(base_dir=os.path.join(self.tmp_dir, 'ltm'))
        self.persistence = SkillPersistence(self.memory)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load_skill_with_named_function(self):
        """命名函数 action 的技能可完整保存与加载。"""
        skill = Skill(
            name='fetch',
            description='获取数据',
            capabilities=['network'],
            action=named_fetch_action,
        )
        skill.update_stats(True)
        skill.update_stats(True)
        skill.update_stats(False)

        result = self.persistence.save_skill(skill)
        self.assertTrue(result['metadata_saved'])
        self.assertTrue(result['action_saved'])

        loaded = self.persistence.load_skill('fetch')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, 'fetch')
        self.assertEqual(loaded.description, '获取数据')
        self.assertEqual(loaded.capabilities, ['network'])
        self.assertEqual(loaded.success_count, 2)
        self.assertEqual(loaded.failure_count, 1)
        # action 应被恢复
        self.assertIsNotNone(loaded.action)

    def test_save_and_load_skill_with_lambda_falls_back_to_metadata(self):
        """lambda action 无法 pickle，应降级为元数据模式。"""
        skill = Skill(
            name='lambda_skill',
            description='lambda 技能',
            capabilities=['cpu'],
            action=lambda t: f"result {t.params.get('x', 0)}",
        )
        result = self.persistence.save_skill(skill)
        self.assertTrue(result['metadata_saved'])
        # lambda 不可 pickle
        self.assertFalse(result['action_saved'])

        loaded = self.persistence.load_skill('lambda_skill')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, 'lambda_skill')
        self.assertEqual(loaded.capabilities, ['cpu'])
        # action 应为 None（降级模式）
        self.assertIsNone(loaded.action)

    def test_save_registry_persists_all_skills(self):
        """保存整个注册中心。"""
        registry = SkillRegistry()
        registry.register(Skill(
            name='fetch', action=named_fetch_action, capabilities=['network'],
        ))
        registry.register(Skill(
            name='normalize', action=named_normalize_action, capabilities=['cpu'],
        ))

        results = self.persistence.save_registry(registry)
        self.assertEqual(len(results), 2)
        saved_names = {r['name'] for r in results}
        self.assertEqual(saved_names, {'fetch', 'normalize'})

    def test_load_into_empty_registry(self):
        """加载到空注册中心应恢复全部技能。"""
        registry = SkillRegistry()
        registry.register(Skill(
            name='fetch', action=named_fetch_action, capabilities=['network'],
        ))
        self.persistence.save_registry(registry)

        new_registry = SkillRegistry()
        loaded = self.persistence.load_into(new_registry)
        self.assertIn('fetch', loaded)
        self.assertIn('fetch', new_registry.list_skills())
        skill = new_registry.get('fetch')
        self.assertIsNotNone(skill)
        self.assertEqual(skill.capabilities, ['network'])

    def test_load_into_existing_registry_no_overwrite(self):
        """不覆盖模式下同名技能应跳过。"""
        registry = SkillRegistry()
        registry.register(Skill(
            name='fetch', action=named_fetch_action, capabilities=['network'],
        ))
        self.persistence.save_registry(registry)

        new_registry = SkillRegistry()
        new_registry.register(Skill(
            name='fetch', description='本地版本', capabilities=['disk'],
        ))
        loaded = self.persistence.load_into(new_registry, overwrite=False)
        # fetch 已存在，不应加载
        self.assertNotIn('fetch', loaded)
        skill = new_registry.get('fetch')
        self.assertEqual(skill.description, '本地版本')

    def test_load_into_existing_registry_with_overwrite(self):
        """覆盖模式下同名技能应被替换。"""
        registry = SkillRegistry()
        registry.register(Skill(
            name='fetch', description='远程版本', action=named_fetch_action,
            capabilities=['network'],
        ))
        self.persistence.save_registry(registry)

        new_registry = SkillRegistry()
        new_registry.register(Skill(
            name='fetch', description='本地版本', capabilities=['disk'],
        ))
        loaded = self.persistence.load_into(new_registry, overwrite=True)
        self.assertIn('fetch', loaded)
        skill = new_registry.get('fetch')
        self.assertEqual(skill.description, '远程版本')

    def test_list_persisted(self):
        """列出已持久化的技能名。"""
        registry = SkillRegistry()
        registry.register(Skill(name='a', capabilities=['x']))
        registry.register(Skill(name='b', capabilities=['y']))
        self.persistence.save_registry(registry)

        persisted = self.persistence.list_persisted()
        self.assertIn('a', persisted)
        self.assertIn('b', persisted)

    def test_list_persisted_empty(self):
        """无持久化技能时返回空列表。"""
        self.assertEqual(self.persistence.list_persisted(), [])

    def test_load_nonexistent_skill(self):
        """加载不存在的技能返回 None。"""
        self.assertIsNone(self.persistence.load_skill('nonexistent'))

    def test_delete_skill(self):
        """删除已持久化的技能。"""
        skill = Skill(name='deletable', capabilities=['x'])
        self.persistence.save_skill(skill)
        self.assertIn('deletable', self.persistence.list_persisted())

        self.assertTrue(self.persistence.delete_skill('deletable'))
        self.assertNotIn('deletable', self.persistence.list_persisted())

    def test_delete_nonexistent_skill(self):
        """删除不存在的技能返回 False。"""
        self.assertFalse(self.persistence.delete_skill('nonexistent'))

    def test_cross_session_transfer(self):
        """端到端：会话 A 保存技能，会话 B 加载并执行。"""
        # 会话 A：注册并保存
        registry_a = SkillRegistry()
        registry_a.register(Skill(
            name='fetch',
            description='获取数据',
            capabilities=['network'],
            action=named_fetch_action,
        ))
        self.persistence.save_registry(registry_a)

        # 会话 B：新建 memory（同路径）并加载
        memory_b = LongTermMemory(base_dir=os.path.join(self.tmp_dir, 'ltm'))
        persistence_b = SkillPersistence(memory_b)
        registry_b = SkillRegistry()
        loaded = persistence_b.load_into(registry_b)
        self.assertIn('fetch', loaded)

        skill = registry_b.get('fetch')
        self.assertIsNotNone(skill)
        self.assertIsNotNone(skill.action)

        # 验证加载的技能可执行
        task = Task(name='fetch', params={'source': 'api.example.com'})
        result = skill.execute(task)
        self.assertTrue(result['success'])
        self.assertIn('api.example.com', result['result'])

    def test_persisted_skill_preserves_stats(self):
        """持久化后技能统计应保留。"""
        skill = Skill(name='tracked', action=named_fetch_action, capabilities=['cpu'])
        for _ in range(7):
            skill.update_stats(True)
        for _ in range(3):
            skill.update_stats(False)
        skill.usage_count = 10

        self.persistence.save_skill(skill)
        loaded = self.persistence.load_skill('tracked')
        self.assertEqual(loaded.success_count, 7)
        self.assertEqual(loaded.failure_count, 3)
        self.assertEqual(loaded.usage_count, 10)
        self.assertAlmostEqual(loaded.success_rate, 0.7)


if __name__ == '__main__':
    unittest.main()
