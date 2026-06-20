"""阶段147：技能自动发现与组合测试。"""

import unittest

from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_composer import SkillDiscovery, SkillComposer
from src.task.task_definition import Task


class TestSkillDiscovery(unittest.TestCase):
    """SkillDiscovery 模式发现测试。"""

    def test_discover_empty_records(self):
        """空记录返回空模式列表。"""
        discovery = SkillDiscovery()
        self.assertEqual(discovery.discover([]), [])

    def test_discover_single_record(self):
        """单条记录无法形成长度>=2的模式。"""
        discovery = SkillDiscovery()
        records = [{'task_name': 'fetch', 'success': True}]
        self.assertEqual(discovery.discover(records), [])

    def test_discover_repeated_pattern(self):
        """重复出现的任务序列应被发现。"""
        discovery = SkillDiscovery(min_support=2, min_length=2)
        records = [
            {'task_name': 'fetch', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'normalize', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'fetch', 'success': True, 'mission_id': 'm2'},
            {'task_name': 'normalize', 'success': True, 'mission_id': 'm2'},
        ]
        patterns = discovery.discover(records)
        pattern_names = [p['pattern'] for p in patterns]
        self.assertIn(['fetch', 'normalize'], pattern_names)
        target = next(p for p in patterns if p['pattern'] == ['fetch', 'normalize'])
        self.assertEqual(target['support'], 2)
        self.assertEqual(target['avg_success'], 1.0)

    def test_discover_respects_min_support(self):
        """低于 min_support 的模式应被过滤。"""
        discovery = SkillDiscovery(min_support=3, min_length=2)
        records = [
            {'task_name': 'a', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'b', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'a', 'success': True, 'mission_id': 'm2'},
            {'task_name': 'b', 'success': True, 'mission_id': 'm2'},
        ]
        patterns = discovery.discover(records)
        self.assertEqual(patterns, [])

    def test_discover_partial_success(self):
        """部分成功的模式 avg_success 应介于 0 和 1 之间。"""
        discovery = SkillDiscovery(min_support=2, min_length=2)
        records = [
            {'task_name': 'a', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'b', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'a', 'success': True, 'mission_id': 'm2'},
            {'task_name': 'b', 'success': False, 'mission_id': 'm2'},
        ]
        patterns = discovery.discover(records)
        target = next(p for p in patterns if p['pattern'] == ['a', 'b'])
        self.assertEqual(target['support'], 2)
        self.assertAlmostEqual(target['avg_success'], 0.5)

    def test_discover_groups_by_mission(self):
        """不同 mission 的任务不应跨组拼接。"""
        discovery = SkillDiscovery(min_support=1, min_length=2)
        records = [
            {'task_name': 'a', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'b', 'success': True, 'mission_id': 'm2'},
        ]
        patterns = discovery.discover(records)
        # a 和 b 属于不同 mission，不应形成 ['a','b'] 模式
        pattern_names = [p['pattern'] for p in patterns]
        self.assertNotIn(['a', 'b'], pattern_names)

    def test_discover_longer_patterns(self):
        """能发现长度为3的频繁模式。"""
        discovery = SkillDiscovery(min_support=2, min_length=2, max_length=3)
        records = []
        for mid in ['m1', 'm2']:
            for name in ['fetch', 'normalize', 'validate']:
                records.append({'task_name': name, 'success': True, 'mission_id': mid})
        patterns = discovery.discover(records)
        pattern_names = [p['pattern'] for p in patterns]
        self.assertIn(['fetch', 'normalize', 'validate'], pattern_names)

    def test_discover_sorted_by_support(self):
        """结果应按 support 降序排列。"""
        discovery = SkillDiscovery(min_support=1, min_length=2)
        records = [
            {'task_name': 'a', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'b', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'a', 'success': True, 'mission_id': 'm2'},
            {'task_name': 'b', 'success': True, 'mission_id': 'm2'},
            {'task_name': 'c', 'success': True, 'mission_id': 'm3'},
            {'task_name': 'd', 'success': True, 'mission_id': 'm3'},
        ]
        patterns = discovery.discover(records)
        supports = [p['support'] for p in patterns]
        self.assertEqual(supports, sorted(supports, reverse=True))


class TestSkillComposer(unittest.TestCase):
    """SkillComposer 技能组合测试。"""

    def setUp(self):
        self.registry = SkillRegistry()
        self.registry.register(Skill(
            name='fetch',
            action=lambda t: f"fetched {t.params.get('source', '')}",
            capabilities=['network'],
        ))
        self.registry.register(Skill(
            name='normalize',
            action=lambda t: f"normalized {t.params.get('records', 0)}",
            capabilities=['cpu'],
        ))
        self.composer = SkillComposer(self.registry)

    def test_compose_creates_composite_skill(self):
        """组合两个技能应生成新的复合技能。"""
        composite = self.composer.compose(['fetch', 'normalize'])
        self.assertIsNotNone(composite)
        self.assertEqual(composite.name, 'fetch+normalize')
        self.assertIn('network', composite.capabilities)
        self.assertIn('cpu', composite.capabilities)

    def test_compose_registers_in_registry(self):
        """复合技能应自动注册到注册中心。"""
        self.composer.compose(['fetch', 'normalize'])
        self.assertIn('fetch+normalize', self.registry.list_skills())

    def test_compose_executes_components_in_order(self):
        """复合技能执行时应按顺序调用组件。"""
        composite = self.composer.compose(['fetch', 'normalize'])
        task = Task(name='pipeline', params={'source': 'api', 'records': 100})
        result = composite.execute(task)
        self.assertTrue(result['success'])
        details = result['result']['details']
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0]['skill'], 'fetch')
        self.assertEqual(details[1]['skill'], 'normalize')

    def test_compose_stops_on_failure(self):
        """组件失败时复合技能应中止并返回失败。"""

        def fail_action(task):
            raise RuntimeError('boom')

        self.registry.register(Skill(
            name='fail_step',
            action=fail_action,
            capabilities=['disk'],
        ))
        composite = self.composer.compose(['fetch', 'fail_step', 'normalize'])
        task = Task(name='pipeline', params={'source': 'api'})
        result = composite.execute(task)
        self.assertFalse(result['success'])
        self.assertIn('fail_step', result['result'])

    def test_compose_missing_component_returns_none(self):
        """组件不存在时应返回 None。"""
        result = self.composer.compose(['fetch', 'nonexistent'])
        self.assertIsNone(result)

    def test_compose_empty_pattern_returns_none(self):
        """空模式应返回 None。"""
        self.assertIsNone(self.composer.compose([]))

    def test_compose_custom_name(self):
        """支持自定义复合技能名。"""
        composite = self.composer.compose(['fetch', 'normalize'], name='data_pipeline')
        self.assertEqual(composite.name, 'data_pipeline')

    def test_compose_union_capabilities_preserves_order(self):
        """能力标签并集应保持首次出现顺序且去重。"""
        self.registry.register(Skill(
            name='validate',
            action=lambda t: 'ok',
            capabilities=['cpu', 'memory'],
        ))
        composite = self.composer.compose(['fetch', 'normalize', 'validate'])
        # fetch: [network], normalize: [cpu], validate: [cpu, memory]
        self.assertEqual(composite.capabilities, ['network', 'cpu', 'memory'])

    def test_compose_from_records_end_to_end(self):
        """从执行记录自动发现并组合复合技能。"""
        records = [
            {'task_name': 'fetch', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'normalize', 'success': True, 'mission_id': 'm1'},
            {'task_name': 'fetch', 'success': True, 'mission_id': 'm2'},
            {'task_name': 'normalize', 'success': True, 'mission_id': 'm2'},
        ]
        composed = self.composer.compose_from_records(records, top_k=3)
        self.assertGreater(len(composed), 0)
        composite = composed[0]
        self.assertIn(composite.name, self.registry.list_skills())
        # 验证复合技能可执行
        task = Task(name='test', params={'source': 'api', 'records': 10})
        result = composite.execute(task)
        self.assertTrue(result['success'])

    def test_compose_from_records_no_patterns(self):
        """无可发现模式时返回空列表。"""
        records = [{'task_name': 'fetch', 'success': True, 'mission_id': 'm1'}]
        composed = self.composer.compose_from_records(records)
        self.assertEqual(composed, [])

    def test_compose_update_existing(self):
        """同名复合技能已存在时应更新而非失败。"""
        self.composer.compose(['fetch', 'normalize'])
        second = self.composer.compose(['fetch', 'normalize'])
        # 应返回技能对象且注册中心仍只有一个同名技能
        self.assertIsNotNone(second)
        names = [s for s in self.registry.list_skills() if s == 'fetch+normalize']
        self.assertEqual(len(names), 1)


if __name__ == '__main__':
    unittest.main()
