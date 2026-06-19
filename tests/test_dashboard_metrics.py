import json
import os
import sys
import time
import unittest
from http.client import HTTPConnection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.server import DashboardServer, DashboardHandler
from src.monitoring.performance_tracker import PerformanceTracker
from src.planning.multi_agent_runner import MultiAgentLongTaskRunner
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.task.task_definition import Task


class TestDashboardMetricsIntegration(unittest.TestCase):
    """Dashboard /api/metrics 与 /api/skills 集成测试"""

    @classmethod
    def setUpClass(cls):
        # 使用非默认端口避免冲突
        cls.server = DashboardServer(host='127.0.0.1', port=18099)
        cls.server.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        # 清理类属性，避免跨测试污染
        DashboardHandler.metrics_provider = None
        DashboardHandler.skills_provider = None

    def setUp(self):
        # 每个测试前重置 provider，避免顺序依赖
        DashboardHandler.metrics_provider = None
        DashboardHandler.skills_provider = None

    def _get(self, path):
        conn = HTTPConnection('127.0.0.1', 18099, timeout=3)
        conn.request('GET', path)
        resp = conn.getresponse()
        body = resp.read().decode('utf-8')
        conn.close()
        return resp.status, body

    def test_metrics_endpoint_without_provider(self):
        status, body = self._get('/api/metrics')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data, {})

    def test_skills_endpoint_without_provider(self):
        status, body = self._get('/api/skills')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data, {})

    def test_metrics_endpoint_with_provider(self):
        tracker = PerformanceTracker()
        tracker.record('mission', 0.5)
        tracker.record('step', 0.01)
        self.server.set_metrics_provider(tracker.summary)

        status, body = self._get('/api/metrics')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn('mission', data)
        self.assertEqual(data['mission']['count'], 1)
        self.assertIn('step', data)

    def test_skills_endpoint_with_provider(self):
        orch = MultiAgentOrchestrator()
        orch.register_worker(AgentWorker(worker_id='w1', capabilities=['cpu']))
        runner = MultiAgentLongTaskRunner(orch)
        runner.register_skill('compute', lambda t: t.params['x'] * 2, ['cpu'])
        self.server.set_skills_provider(runner.skill_stats)

        status, body = self._get('/api/skills')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data['skill_count'], 1)
        self.assertEqual(data['skills'][0]['name'], 'compute')

    def test_metrics_reflects_real_execution(self):
        orch = MultiAgentOrchestrator()
        orch.register_worker(AgentWorker(worker_id='w1', capabilities=['cpu']))
        tracker = PerformanceTracker()
        runner = MultiAgentLongTaskRunner(orch, tracker=tracker)
        runner.register_skill('compute', lambda t: t.params['x'] + 1, ['cpu'])

        goal = Task(name='metrics_goal')
        steps = [Task(name='compute', params={'required_capabilities': ['cpu'], 'x': i}) for i in range(5)]
        runner.run(goal, steps)

        self.server.set_metrics_provider(runner.performance_summary)
        status, body = self._get('/api/metrics')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data['step']['count'], 5)
        self.assertEqual(data['mission']['count'], 1)


if __name__ == '__main__':
    unittest.main()
