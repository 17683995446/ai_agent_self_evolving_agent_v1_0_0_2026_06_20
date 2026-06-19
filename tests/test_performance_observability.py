import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.performance_tracker import PerformanceTracker
from src.planning.multi_agent_runner import MultiAgentLongTaskRunner
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.task.task_definition import Task


class TestPerformanceTracker(unittest.TestCase):
    """PerformanceTracker 单元测试"""

    def test_record_and_summary(self):
        tracker = PerformanceTracker()
        tracker.record('op_a', 0.1)
        tracker.record('op_a', 0.3)
        tracker.record('op_b', 0.2)
        summary = tracker.summary()
        self.assertEqual(summary['op_a']['count'], 2)
        self.assertAlmostEqual(summary['op_a']['mean'], 0.2, places=5)
        self.assertEqual(summary['op_b']['count'], 1)
        self.assertIn('p95', summary['op_a'])

    def test_time_context_manager(self):
        tracker = PerformanceTracker()
        with tracker.time('block', tags={'name': 'sleep'}):
            time.sleep(0.01)
        summary = tracker.summary()
        self.assertEqual(summary['block']['count'], 1)
        self.assertGreaterEqual(summary['block']['mean'], 0.0)

    def test_percentile(self):
        tracker = PerformanceTracker()
        for i in range(1, 101):
            tracker.record('linear', float(i) / 1000.0)
        summary = tracker.summary()
        self.assertAlmostEqual(summary['linear']['p95'], 0.09505, places=5)


class TestMultiAgentPerformance(unittest.TestCase):
    """多 Agent 长任务性能基准测试"""

    def setUp(self):
        self.orchestrator = MultiAgentOrchestrator(strategy='capability_match')
        self.orchestrator.register_worker(AgentWorker(worker_id='w1', capabilities=['cpu']))
        self.orchestrator.register_worker(AgentWorker(worker_id='w2', capabilities=['disk']))

    def test_mission_latency_under_threshold(self):
        tracker = PerformanceTracker()
        runner = MultiAgentLongTaskRunner(self.orchestrator, tracker=tracker)
        runner.register_skill('compute', lambda t: t.params['x'] * 2, ['cpu'])
        runner.register_skill('save', lambda t: f"saved {t.params['x']}", ['disk'])

        goal = Task(name='benchmark_pipeline')
        steps = []
        for i in range(50):
            cap = 'cpu' if i % 2 == 0 else 'disk'
            steps.append(Task(name='compute' if cap == 'cpu' else 'save', params={'required_capabilities': [cap], 'x': i}))

        summary = runner.run(goal, steps)
        self.assertEqual(summary['status'], 'completed')

        perf = runner.performance_summary()
        self.assertIn('mission', perf)
        self.assertIn('step', perf)
        self.assertEqual(perf['step']['count'], 50)
        # 50 步轻量任务整体应远小于 2 秒
        self.assertLess(perf['mission']['max'], 2.0)

    def test_skill_registry_query_performance(self):
        registry = SkillRegistry()
        for i in range(100):
            registry.register(Skill(name=f'skill_{i}', capabilities=['cpu']))
        task = Task(name='job', params={'required_capabilities': ['cpu']})
        start = time.time()
        for _ in range(1000):
            registry.select_best(task)
        elapsed = time.time() - start
        # 1000 次选择应在 1 秒内完成
        self.assertLess(elapsed, 1.0)


if __name__ == '__main__':
    unittest.main()
