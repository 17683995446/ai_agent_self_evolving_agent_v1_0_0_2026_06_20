"""阶段151：自我评估与能力差距分析测试。"""

import unittest

from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.skills.capability_assessor import CapabilityAssessor, GapAnalyzer


class TestCapabilityAssessor(unittest.TestCase):
    """CapabilityAssessor 能力评估测试。"""

    def setUp(self):
        self.registry = SkillRegistry()

    def test_assess_empty_registry(self):
        """空注册表评估返回零值。"""
        assessor = CapabilityAssessor()
        result = assessor.assess(self.registry)
        self.assertEqual(result['skill_count'], 0)
        self.assertEqual(result['skill_capabilities'], [])
        self.assertEqual(result['avg_success_rate'], 0.0)

    def test_assess_collects_capabilities(self):
        """评估应收集全部技能能力标签。"""
        self.registry.register(Skill(name='a', capabilities=['cpu', 'network']))
        self.registry.register(Skill(name='b', capabilities=['cpu', 'disk']))
        assessor = CapabilityAssessor()
        result = assessor.assess(self.registry)
        self.assertEqual(set(result['skill_capabilities']), {'cpu', 'network', 'disk'})

    def test_assess_worker_capabilities(self):
        """评估应合并 worker 能力。"""
        self.registry.register(Skill(name='a', capabilities=['cpu']))
        assessor = CapabilityAssessor()
        result = assessor.assess(
            self.registry,
            worker_capabilities=[['network'], ['disk', 'cpu']],
        )
        self.assertEqual(set(result['worker_capabilities']), {'network', 'disk', 'cpu'})
        self.assertEqual(set(result['total_capabilities']), {'cpu', 'network', 'disk'})

    def test_assess_capability_coverage(self):
        """评估应统计每个能力的技能覆盖数。"""
        self.registry.register(Skill(name='a', capabilities=['cpu', 'network']))
        self.registry.register(Skill(name='b', capabilities=['cpu']))
        assessor = CapabilityAssessor()
        result = assessor.assess(self.registry)
        self.assertEqual(result['capability_coverage']['cpu'], 2)
        self.assertEqual(result['capability_coverage']['network'], 1)

    def test_assess_underperforming_skills(self):
        """低效技能应被识别。"""
        skill = Skill(name='bad', capabilities=['cpu'])
        for _ in range(5):
            skill.update_stats(False)
        skill.usage_count = 5
        self.registry.register(skill)

        good = Skill(name='good', capabilities=['cpu'])
        for _ in range(5):
            good.update_stats(True)
        good.usage_count = 5
        self.registry.register(good)

        assessor = CapabilityAssessor(underperform_threshold=0.3)
        result = assessor.assess(self.registry)
        self.assertIn('bad', result['underperforming_skills'])
        self.assertNotIn('good', result['underperforming_skills'])

    def test_assess_unused_skills(self):
        """未使用技能应被识别。"""
        self.registry.register(Skill(name='unused', capabilities=['cpu']))
        used = Skill(name='used', capabilities=['cpu'])
        used.usage_count = 3
        self.registry.register(used)

        assessor = CapabilityAssessor(unused_threshold=0)
        result = assessor.assess(self.registry)
        self.assertIn('unused', result['unused_skills'])
        self.assertNotIn('used', result['unused_skills'])

    def test_assess_proficient_skills(self):
        """高效技能应被识别。"""
        skill = Skill(name='expert', capabilities=['cpu'])
        for _ in range(5):
            skill.update_stats(True)
        skill.usage_count = 5
        self.registry.register(skill)

        assessor = CapabilityAssessor()
        result = assessor.assess(self.registry)
        self.assertIn('expert', result['proficient_skills'])

    def test_assess_avg_success_rate(self):
        """平均成功率应正确计算。"""
        s1 = Skill(name='a', capabilities=['cpu'])
        s1.update_stats(True)
        s1.update_stats(False)
        s1.usage_count = 2
        self.registry.register(s1)

        s2 = Skill(name='b', capabilities=['cpu'])
        s2.update_stats(True)
        s2.update_stats(True)
        s2.usage_count = 2
        self.registry.register(s2)

        assessor = CapabilityAssessor()
        result = assessor.assess(self.registry)
        # s1: 0.5, s2: 1.0 -> avg 0.75
        self.assertAlmostEqual(result['avg_success_rate'], 0.75)


class TestGapAnalyzer(unittest.TestCase):
    """GapAnalyzer 能力差距分析测试。"""

    def setUp(self):
        self.assessment = {
            'skill_count': 3,
            'skill_capabilities': ['cpu', 'network'],
            'worker_capabilities': ['disk'],
            'total_capabilities': ['cpu', 'disk', 'network'],
            'capability_coverage': {'cpu': 2, 'network': 1, 'disk': 0},
            'avg_success_rate': 0.6,
            'underperforming_skills': ['bad_skill'],
            'unused_skills': ['unused_skill'],
            'proficient_skills': ['good_skill'],
        }
        self.analyzer = GapAnalyzer(min_coverage=1)

    def test_analyze_no_gap(self):
        """所有需求能力均已覆盖时无差距。"""
        result = self.analyzer.analyze(
            self.assessment,
            required_capabilities=['cpu', 'network'],
        )
        self.assertEqual(result['missing_capabilities'], [])
        self.assertEqual(result['gap_count'], 0)

    def test_analyze_missing_capabilities(self):
        """应识别缺失能力。"""
        result = self.analyzer.analyze(
            self.assessment,
            required_capabilities=['cpu', 'gpu'],
        )
        self.assertIn('gpu', result['missing_capabilities'])
        self.assertNotIn('cpu', result['missing_capabilities'])

    def test_analyze_understaffed_capabilities(self):
        """应识别覆盖不足能力。"""
        analyzer = GapAnalyzer(min_coverage=2)
        result = analyzer.analyze(
            self.assessment,
            required_capabilities=['cpu', 'network'],
        )
        # cpu 有 2 个技能覆盖，network 只有 1 个
        self.assertIn('network', result['understaffed_capabilities'])
        self.assertNotIn('cpu', result['understaffed_capabilities'])

    def test_analyze_generates_create_skill_suggestions(self):
        """缺失能力应生成创建技能建议。"""
        result = self.analyzer.analyze(
            self.assessment,
            required_capabilities=['gpu'],
        )
        create_suggestions = [s for s in result['suggestions'] if s['type'] == 'create_skill']
        self.assertEqual(len(create_suggestions), 1)
        self.assertEqual(create_suggestions[0]['capability'], 'gpu')
        self.assertEqual(create_suggestions[0]['priority'], 'high')

    def test_analyze_generates_practice_suggestions(self):
        """低效技能应生成练习建议。"""
        result = self.analyzer.analyze(
            self.assessment,
            required_capabilities=['cpu'],
        )
        practice_suggestions = [s for s in result['suggestions'] if s['type'] == 'practice_skill']
        self.assertEqual(len(practice_suggestions), 1)
        self.assertEqual(practice_suggestions[0]['skill_name'], 'bad_skill')

    def test_analyze_generates_explore_suggestions(self):
        """未使用技能应生成探索建议。"""
        result = self.analyzer.analyze(
            self.assessment,
            required_capabilities=['cpu'],
        )
        explore_suggestions = [s for s in result['suggestions'] if s['type'] == 'explore_skill']
        self.assertEqual(len(explore_suggestions), 1)
        self.assertEqual(explore_suggestions[0]['skill_name'], 'unused_skill')

    def test_suggest_tasks_from_gap_analysis(self):
        """差距分析应能转化为可执行任务建议。"""
        result = self.analyzer.analyze(
            self.assessment,
            required_capabilities=['gpu'],
        )
        tasks = self.analyzer.suggest_tasks(result)
        self.assertGreater(len(tasks), 0)
        create_tasks = [t for t in tasks if 'learn_' in t['task_name']]
        self.assertEqual(len(create_tasks), 1)
        self.assertEqual(create_tasks[0]['capability'], 'gpu')

    def test_suggest_tasks_includes_practice(self):
        """任务建议应包含练习任务。"""
        result = self.analyzer.analyze(
            self.assessment,
            required_capabilities=['cpu'],
        )
        tasks = self.analyzer.suggest_tasks(result)
        practice_tasks = [t for t in tasks if 'practice_' in t['task_name']]
        self.assertEqual(len(practice_tasks), 1)

    def test_end_to_end_assessment_and_gap_analysis(self):
        """端到端：评估 → 差距分析 → 任务建议。"""
        registry = SkillRegistry()
        registry.register(Skill(name='fetch', capabilities=['network']))
        registry.register(Skill(name='process', capabilities=['cpu']))

        assessor = CapabilityAssessor()
        assessment = assessor.assess(
            registry,
            worker_capabilities=[['disk']],
        )

        analyzer = GapAnalyzer(min_coverage=1)
        # 任务需要 gpu 但无人覆盖
        gap = analyzer.analyze(assessment, required_capabilities=['network', 'cpu', 'gpu'])
        tasks = analyzer.suggest_tasks(gap)

        self.assertIn('gpu', gap['missing_capabilities'])
        gpu_tasks = [t for t in tasks if t['capability'] == 'gpu']
        self.assertEqual(len(gpu_tasks), 1)
        self.assertEqual(gpu_tasks[0]['priority'], 'high')


if __name__ == '__main__':
    unittest.main()
