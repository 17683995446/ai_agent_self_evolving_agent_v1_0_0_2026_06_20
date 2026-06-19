#!/usr/bin/env python3
"""
仪表盘与监控接口健壮性测试
目标：验证DashboardServer在高并发下稳定，JSON格式一致，指标计算准确
"""

import os
import sys
import unittest
import threading
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.server import DashboardServer, DashboardDataStore


class TestDashboardDataStore(unittest.TestCase):
    """仪表盘数据存储测试"""

    def test_update_basic(self):
        ds = DashboardDataStore()
        ds.update({
            'cycle': 1,
            'state': {'self': {}},
            'safety': {'emergency_mode': False},
            'predictions': {'low': {'cpu': 0.5}},
            'errors': {'cpu': 0.1},
            'reward': {'total_reward': 0.8},
            'duration_seconds': 0.5,
        })
        data = ds.get_data()
        self.assertEqual(data['cycles'], 1)
        self.assertEqual(data['current_state'], 'ACTIVE')
        self.assertEqual(data['reward'], 0.8)

    def test_update_emergency_state(self):
        ds = DashboardDataStore()
        ds.update({
            'cycle': 2,
            'safety': {'emergency_mode': True},
            'errors': {},
            'reward': {'total_reward': 0.0},
        })
        data = ds.get_data()
        self.assertEqual(data['current_state'], 'EMERGENCY')

    def test_history_bounded(self):
        ds = DashboardDataStore()
        for i in range(150):
            ds.update({
                'cycle': i,
                'safety': {'emergency_mode': False},
                'errors': {'cpu': 0.1},
                'reward': {'total_reward': 0.5},
            })
        data = ds.get_data()
        self.assertLessEqual(len(data['history']), 100)

    def test_history_mae_calculation(self):
        ds = DashboardDataStore()
        ds.update({
            'cycle': 1,
            'safety': {'emergency_mode': False},
            'errors': {'cpu': 0.1, 'mem': 0.3},
            'reward': {'total_reward': 0.5},
        })
        data = ds.get_data()
        self.assertAlmostEqual(data['history'][0]['mae'], 0.2, places=6)

    def test_update_stats(self):
        ds = DashboardDataStore()
        ds.update_stats({
            'recent_mae': 0.05,
            'recent_success_rate': 0.9,
            'task_pool': {'total_tasks': 10},
            'memory_samples': 100,
        })
        data = ds.get_data()
        self.assertEqual(data['recent_mae'], 0.05)
        self.assertEqual(data['success_rate'], 0.9)
        self.assertEqual(data['task_pool']['total_tasks'], 10)
        self.assertEqual(data['memory_samples'], 100)

    def test_concurrent_update(self):
        """多线程并发更新不应崩溃"""
        ds = DashboardDataStore()
        errors = []

        def updater():
            for i in range(50):
                try:
                    ds.update({
                        'cycle': i,
                        'safety': {'emergency_mode': False},
                        'errors': {'cpu': 0.1},
                        'reward': {'total_reward': 0.5},
                    })
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=updater) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        data = ds.get_data()
        self.assertGreaterEqual(data['cycles'], 0)

    def test_safe_json_dumps_with_unserializable(self):
        """不可序列化对象应被安全处理"""
        ds = DashboardDataStore()
        ds._data['test_object'] = set([1, 2, 3])
        result = ds._safe_json_dumps(ds._data)
        self.assertIsInstance(result, bytes)
        parsed = json.loads(result.decode('utf-8'))
        self.assertIn('test_object', parsed)


class TestDashboardServer(unittest.TestCase):
    """仪表盘服务器测试"""

    def setUp(self):
        self.server = DashboardServer(host='127.0.0.1', port=0)  # port=0让系统分配端口

    def tearDown(self):
        if self.server.is_running():
            self.server.stop()

    def _get_port(self):
        """获取实际分配的端口"""
        if self.server._server:
            return self.server._server.server_address[1]
        return None

    def _fetch(self, path):
        port = self._get_port()
        url = f'http://127.0.0.1:{port}{path}'
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.read().decode('utf-8')

    def test_server_start_stop(self):
        """启动和停止"""
        self.server.start()
        self.assertTrue(self.server.is_running())
        self.server.stop()
        self.assertFalse(self.server.is_running())

    def test_health_endpoint(self):
        """健康检查接口"""
        self.server.start()
        response = self._fetch('/api/health')
        data = json.loads(response)
        self.assertEqual(data['status'], 'ok')
        self.server.stop()

    def test_data_endpoint(self):
        """数据接口"""
        self.server.start()
        # 先更新一些数据
        self.server.update({
            'cycle': 42,
            'safety': {'emergency_mode': False},
            'predictions': {'low': {'cpu': 0.5}},
            'errors': {'cpu': 0.1},
            'reward': {'total_reward': 0.8},
            'duration_seconds': 0.5,
        })
        self.server.update_stats({
            'recent_mae': 0.05,
            'recent_success_rate': 0.9,
            'task_pool': {'total_tasks': 5},
            'memory_samples': 100,
        })

        response = self._fetch('/api/data')
        data = json.loads(response)
        self.assertEqual(data['cycles'], 42)
        self.assertEqual(data['current_state'], 'ACTIVE')
        self.assertEqual(data['reward'], 0.8)
        self.assertEqual(data['memory_samples'], 100)
        self.server.stop()

    def test_html_endpoint(self):
        """HTML页面接口"""
        self.server.start()
        response = self._fetch('/')
        self.assertIn('<html', response.lower())
        self.assertIn('自驱', response)
        self.server.stop()

    def test_404_endpoint(self):
        """不存在的路径应返回404"""
        self.server.start()
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._fetch('/nonexistent')
        self.assertEqual(cm.exception.code, 404)
        self.server.stop()

    def test_concurrent_requests(self):
        """并发请求不应崩溃"""
        self.server.start()
        self.server.update({
            'cycle': 1,
            'safety': {'emergency_mode': False},
            'errors': {},
            'reward': {'total_reward': 0.5},
        })

        errors = []
        results = []

        def fetcher():
            try:
                for _ in range(10):
                    r = self._fetch('/api/data')
                    results.append(r)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=fetcher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.server.stop()
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 50)

    def test_server_runs_without_agent(self):
        """仪表盘应能独立运行（不依赖AgentLoop）"""
        self.server.start()
        # 不调用update，直接获取数据
        response = self._fetch('/api/data')
        data = json.loads(response)
        self.assertIn('cycles', data)
        self.assertIn('history', data)
        self.server.stop()

    def test_data_isolation_after_stop(self):
        """停止后数据应保留"""
        self.server.start()
        self.server.update({
            'cycle': 99,
            'safety': {'emergency_mode': False},
            'errors': {},
            'reward': {'total_reward': 0.9},
        })
        self.server.stop()

        # 停止后data_store仍应保留数据
        data = self.server.data_store.get_data()
        self.assertEqual(data['cycles'], 99)


if __name__ == '__main__':
    unittest.main()
