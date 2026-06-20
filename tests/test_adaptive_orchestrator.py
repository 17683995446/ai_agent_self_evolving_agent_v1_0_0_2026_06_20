"""阶段149：自适应编排策略学习测试。"""

import unittest

from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.multi_agent.adaptive_orchestrator import (
    AdaptiveOrchestrator,
    StrategyLearner,
    task_signature,
)
from src.task.task_definition import Task


class TestTaskSignature(unittest.TestCase):
    """任务签名提取测试。"""

    def test_signature_with_capabilities(self):
        """有能力需求时返回排序后的能力集合。"""
        task = Task(name='t', params={'required_capabilities': ['cpu', 'network']})
        self.assertEqual(task_signature(task), 'cpu,network')

    def test_signature_unordered(self):
        """能力顺序不影响签名。"""
        t1 = Task(name='t', params={'required_capabilities': ['network', 'cpu']})
        t2 = Task(name='t', params={'required_capabilities': ['cpu', 'network']})
        self.assertEqual(task_signature(t1), task_signature(t2))

    def test_signature_no_capabilities(self):
        """无能力需求时返回通用签名。"""
        task = Task(name='t', params={})
        self.assertEqual(task_signature(task), '__generic__')


class TestStrategyLearner(unittest.TestCase):
    """StrategyLearner 学习与推荐测试。"""

    def setUp(self):
        self.learner = StrategyLearner(min_samples=3)

    def test_record_and_success_rate(self):
        """记录后应能查询成功率。"""
        self.learner.record('cpu', 'round_robin', True)
        self.learner.record('cpu', 'round_robin', False)
        self.assertAlmostEqual(self.learner.success_rate('cpu', 'round_robin'), 0.5)

    def test_success_rate_no_data(self):
        """无数据时成功率为 0。"""
        self.assertEqual(self.learner.success_rate('cpu', 'round_robin'), 0.0)

    def test_recommend_insufficient_samples(self):
        """采样不足时推荐 None。"""
        self.learner.record('cpu', 'round_robin', True)
        self.learner.record('cpu', 'round_robin', True)
        # 仅 2 次，低于 min_samples=3
        self.assertIsNone(self.learner.recommend('cpu', ['round_robin', 'capability_match']))

    def test_recommend_best_strategy(self):
        """推荐成功率最高的策略。"""
        for _ in range(5):
            self.learner.record('cpu', 'round_robin', True)
        for _ in range(3):
            self.learner.record('cpu', 'capability_match', False)
        recommended = self.learner.recommend('cpu', ['round_robin', 'capability_match'])
        self.assertEqual(recommended, 'round_robin')

    def test_recommend_no_data_returns_none(self):
        """完全无数据时推荐 None。"""
        self.assertIsNone(self.learner.recommend('cpu', ['round_robin']))

    def test_needs_exploration_with_unsampled(self):
        """有未采样策略时返回该策略名。"""
        # round_robin 已充分采样
        for _ in range(5):
            self.learner.record('cpu', 'round_robin', True)
        result = self.learner.needs_exploration('cpu', ['round_robin', 'capability_match'])
        self.assertEqual(result, 'capability_match')

    def test_needs_exploration_all_sampled(self):
        """所有策略均已充分采样时返回 None。"""
        for _ in range(5):
            self.learner.record('cpu', 'round_robin', True)
            self.learner.record('cpu', 'capability_match', True)
        self.assertIsNone(self.learner.needs_exploration('cpu', ['round_robin', 'capability_match']))

    def test_get_stats(self):
        """获取学习统计。"""
        self.learner.record('cpu', 'round_robin', True)
        self.learner.record('cpu', 'round_robin', False)
        stats = self.learner.get_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]['signature'], 'cpu')
        self.assertEqual(stats[0]['strategy'], 'round_robin')
        self.assertEqual(stats[0]['total'], 2)
        self.assertEqual(stats[0]['successes'], 1)

    def test_reset(self):
        """清空学习统计。"""
        self.learner.record('cpu', 'round_robin', True)
        self.learner.reset()
        self.assertEqual(self.learner.get_stats(), [])


class TestAdaptiveOrchestrator(unittest.TestCase):
    """AdaptiveOrchestrator 自适应编排测试。"""

    def setUp(self):
        self.orch = MultiAgentOrchestrator(strategy='round_robin')
        self.orch.register_worker(AgentWorker(worker_id='w1', capabilities=['cpu', 'network']))
        self.orch.register_worker(AgentWorker(worker_id='w2', capabilities=['cpu', 'disk']))
        self.adaptive = AdaptiveOrchestrator(self.orch, default_strategy='round_robin')

    def test_assign_task_auto_selects_strategy(self):
        """未指定策略时自动选择。"""
        task = Task(name='t', params={'required_capabilities': ['cpu']})
        result = self.adaptive.assign_task(task)
        self.assertIsNotNone(result)
        self.assertIsNotNone(self.adaptive.last_strategy)

    def test_assign_task_records_feedback(self):
        """执行后应记录学习反馈。"""
        task = Task(name='t', params={'required_capabilities': ['cpu']})
        self.adaptive.assign_task(task)
        stats = self.adaptive.learner.get_stats()
        self.assertEqual(len(stats), 1)

    def test_explores_unsampled_strategies(self):
        """有未采样策略时应优先探索。"""
        task = Task(name='t', params={'required_capabilities': ['cpu']})
        strategies = self.orch.supported_strategies()
        # 第一次执行应探索第一个未采样策略
        self.adaptive.assign_task(task)
        first_strategy = self.adaptive.last_strategy
        self.assertIn(first_strategy, strategies)

    def test_uses_learned_best_strategy(self):
        """充分采样后应使用历史最优策略。"""
        sig = 'cpu'
        # 让 round_robin 成功率高
        for _ in range(5):
            self.adaptive.learner.record(sig, 'round_robin', True)
        # 让 capability_match 成功率低
        for _ in range(5):
            self.adaptive.learner.record(sig, 'capability_match', False)
        # 让 least_loaded 也有足够采样
        for _ in range(5):
            self.adaptive.learner.record(sig, 'least_loaded', False)

        task = Task(name='t', params={'required_capabilities': ['cpu']})
        self.adaptive.assign_task(task)
        self.assertEqual(self.adaptive.last_strategy, 'round_robin')

    def test_explicit_strategy_overrides_auto(self):
        """显式指定策略时不自动选择。"""
        task = Task(name='t', params={'required_capabilities': ['cpu']})
        self.adaptive.assign_task(task, strategy='least_loaded')
        self.assertEqual(self.adaptive.last_strategy, 'least_loaded')

    def test_run_mission_records_all_tasks(self):
        """run_mission 应为每个任务记录学习反馈。"""
        tasks = [
            Task(name='a', params={'required_capabilities': ['cpu']}),
            Task(name='b', params={'required_capabilities': ['network']}),
        ]
        summary = self.adaptive.run_mission(tasks)
        self.assertEqual(summary['total_tasks'], 2)
        # 应记录了至少 2 条学习反馈
        stats = self.adaptive.learner.get_stats()
        signatures = {s['signature'] for s in stats}
        self.assertIn('cpu', signatures)
        self.assertIn('network', signatures)

    def test_adaptive_learns_from_failures(self):
        """从失败中学习：失败策略的推荐优先级降低。"""
        sig = 'network'
        learner = StrategyLearner(min_samples=2)
        adaptive = AdaptiveOrchestrator(self.orch, learner=learner)

        # 让 round_robin 在 network 任务上全部失败
        for _ in range(3):
            learner.record(sig, 'round_robin', False)
        # 让 capability_match 全部成功
        for _ in range(3):
            learner.record(sig, 'capability_match', True)
        # least_loaded 也采样
        for _ in range(3):
            learner.record(sig, 'least_loaded', False)

        task = Task(name='t', params={'required_capabilities': ['network']})
        adaptive.assign_task(task)
        self.assertEqual(adaptive.last_strategy, 'capability_match')


if __name__ == '__main__':
    unittest.main()
