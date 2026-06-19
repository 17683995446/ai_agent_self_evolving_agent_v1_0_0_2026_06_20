import os
import sys
import unittest
import tempfile
import shutil
import time
import urllib.request
import json
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.server import DashboardServer, DashboardDataStore


class TestDashboardDataStore(unittest.TestCase):
    def test_initial_data(self):
        store = DashboardDataStore()
        data = store.get_data()
        self.assertEqual(data['cycles'], 0)
        self.assertEqual(data['recent_mae'], 0.0)
        self.assertEqual(data['success_rate'], 0.0)

    def test_update(self):
        store = DashboardDataStore()
        store.update({
            'cycle': 5,
            'state': {'self': {'state': 'ACTIVE'}},
            'predictions': {'low': {'cpu': 0.5}},
            'errors': {'cpu': 0.1},
            'reward': {'total_reward': 0.8},
            'duration_seconds': 0.5,
        })

        data = store.get_data()
        self.assertEqual(data['cycles'], 5)
        self.assertEqual(data['current_state'], 'ACTIVE')
        self.assertEqual(data['reward'], 0.8)
        self.assertEqual(len(data['history']), 1)

    def test_history_limit(self):
        store = DashboardDataStore()
        store._max_history = 5

        for i in range(10):
            store.update({
                'cycle': i,
                'state': {'self': {}},
                'predictions': {},
                'errors': {},
                'reward': {'total_reward': 0.5},
                'duration_seconds': 0.1,
            })

        data = store.get_data()
        self.assertEqual(len(data['history']), 5)

    def test_update_stats(self):
        store = DashboardDataStore()
        store.update_stats({
            'recent_mae': 0.05,
            'recent_success_rate': 0.9,
            'task_pool': {'total_tasks': 10},
            'memory_samples': 100,
        })

        data = store.get_data()
        self.assertEqual(data['recent_mae'], 0.05)
        self.assertEqual(data['success_rate'], 0.9)
        self.assertEqual(data['task_pool']['total_tasks'], 10)
        self.assertEqual(data['memory_samples'], 100)

    def test_thread_safety(self):
        store = DashboardDataStore()
        errors = []

        def writer():
            try:
                for i in range(100):
                    store.update({
                        'cycle': i,
                        'state': {'self': {}},
                        'predictions': {},
                        'errors': {},
                        'reward': {'total_reward': 0.5},
                        'duration_seconds': 0.1,
                    })
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    store.get_data()
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=writer)
            t2 = threading.Thread(target=reader)
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestDashboardServer(unittest.TestCase):
    def setUp(self):
        self.server = DashboardServer(host='127.0.0.1', port=0)
        self.server.start()
        # 获取实际分配的端口
        self.port = self.server._server.server_address[1]
        self.base_url = f'http://127.0.0.1:{self.port}'
        time.sleep(0.2)

    def tearDown(self):
        self.server.stop()

    def test_health_endpoint(self):
        req = urllib.request.Request(f'{self.base_url}/api/health')
        with urllib.request.urlopen(req, timeout=2) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data['status'], 'ok')

    def test_data_endpoint(self):
        req = urllib.request.Request(f'{self.base_url}/api/data')
        with urllib.request.urlopen(req, timeout=2) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn('cycles', data)
            self.assertIn('history', data)

    def test_html_endpoint(self):
        req = urllib.request.Request(f'{self.base_url}/')
        with urllib.request.urlopen(req, timeout=2) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode()
            self.assertIn('自驱Agent监控仪表盘', html)

    def test_404(self):
        req = urllib.request.Request(f'{self.base_url}/notfound')
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2)
        self.assertEqual(ctx.exception.code, 404)

    def test_update_integration(self):
        self.server.update({
            'cycle': 42,
            'state': {'self': {'state': 'ACTIVE'}},
            'predictions': {'low': {'cpu': 0.3}},
            'errors': {'cpu': 0.05},
            'reward': {'total_reward': 0.95},
            'duration_seconds': 0.2,
        })

        req = urllib.request.Request(f'{self.base_url}/api/data')
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            self.assertEqual(data['cycles'], 42)
            self.assertEqual(data['reward'], 0.95)
            self.assertEqual(data['current_state'], 'ACTIVE')

    def test_is_running(self):
        self.assertTrue(self.server.is_running())
        self.server.stop()
        self.assertFalse(self.server.is_running())


if __name__ == '__main__':
    unittest.main()
