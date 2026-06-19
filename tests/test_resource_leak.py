#!/usr/bin/env python3
"""
性能回归测试与资源泄漏检测
目标：验证长时间运行后无内存泄漏、无句柄泄漏、无线程泄漏
"""

import os
import sys
import gc
import tempfile
import shutil
import threading
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import patch

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.dashboard.server import DashboardServer
from src.task.task_pool import TaskPool
from src.task.task_definition import Task


class TestResourceLeak(unittest.TestCase):
    """资源泄漏检测"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        self.temp_dir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        for d in ['data', 'logs', 'snapshots']:
            os.makedirs(d, exist_ok=True)
        with open('config.yaml', 'w') as f:
            f.write("""
goals:
  precision_evolution: true
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: 100.0
  max_mem_percent: 100.0
exploration:
  base_ratio: 0.3
safety:
  max_consecutive_failures: 5
review:
  interval_hours: 999
  min_samples: 99999
memory:
  max_short_term_samples: 200
logging:
  level: INFO
  file: logs/agent.log
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
dashboard:
  enabled: false
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_loop_memory_stable(self):
        """Agent运行100个周期后内存增长应可控"""
        agent = AgentLoop('config.yaml')
        gc.collect()
        tracemalloc.start()
        snap1 = tracemalloc.take_snapshot()

        for _ in range(100):
            agent.run_cycle()

        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        agent.stop()

        top_stats = snap2.compare_to(snap1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        self.assertLess(total_diff, 5 * 1024 * 1024, f"Memory leak detected: {total_diff} bytes")

    def test_short_term_memory_db_not_leak(self):
        """SQLite连接不泄漏"""
        db_path = os.path.join(self.temp_dir, 'data', 'test.db')
        memories = []
        for _ in range(10):
            mem = ShortTermMemory(db_path=db_path, max_samples=50)
            mem.store_sample({'timestamp': 1.0, 'state': {}, 'predictions': {}, 'actuals': {}, 'errors': {}, 'reward': 0.0})
            mem.close()
            memories.append(mem)

        active_threads = threading.enumerate()
        mem_threads = [t for t in active_threads if 'sqlite' in t.name.lower()]
        self.assertEqual(len(mem_threads), 0, "SQLite threads leaked")

    def test_dashboard_server_thread_cleanup(self):
        """Dashboard服务器线程应正确清理"""
        server = DashboardServer(host='127.0.0.1', port=0)
        server.start()
        self.assertTrue(server.is_running())
        server.stop()
        self.assertFalse(server.is_running())
        if server._thread:
            self.assertFalse(server._thread.is_alive(), "Dashboard thread leaked")

    def test_task_pool_no_unbounded_growth(self):
        """任务池不应无限增长"""
        pool = TaskPool(max_size=50)
        for i in range(200):
            pool.update([Task(id=f'task_{i}', name=f'task_{i}')])
        stats = pool.get_stats()
        self.assertLessEqual(stats['total_tasks'], 50, "Task pool unbounded growth")

    def test_agent_stop_releases_resources(self):
        """Agent停止后应释放核心资源"""
        agent = AgentLoop('config.yaml')
        for _ in range(10):
            agent.run_cycle()
        agent.stop()

        self.assertFalse(agent._running)
        self.assertTrue(getattr(agent.memory, '_closed', False), "Memory not closed")


if __name__ == '__main__':
    unittest.main()
