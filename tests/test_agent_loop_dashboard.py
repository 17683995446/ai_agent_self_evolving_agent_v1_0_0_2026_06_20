import os
import sys
import unittest
import tempfile
import shutil
import time
import urllib.request
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop


class TestAgentLoopDashboard(unittest.TestCase):
    def setUp(self):
        # 重置Config单例
        from src.core.config import Config
        Config._instance = None
        Config._config = {}

        self.temp_dir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)

        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('snapshots', exist_ok=True)

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
  interval_hours: 6
  min_samples: 100
memory:
  max_short_term_samples: 100
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
dashboard:
  enabled: true
  host: 127.0.0.1
  port: 0
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dashboard_initialized(self):
        agent = AgentLoop('config.yaml')
        self.assertIsNotNone(agent.dashboard)
        agent.stop()

    def test_dashboard_updates_after_cycle(self):
        agent = AgentLoop('config.yaml')
        agent.dashboard.start()
        port = agent.dashboard._server.server_address[1]

        try:
            # 运行一个周期
            agent.run_cycle()

            # 等待数据更新
            time.sleep(0.3)

            # 通过HTTP获取数据
            req = urllib.request.Request(f'http://127.0.0.1:{port}/api/data')
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(data['cycles'], 1)
                self.assertGreaterEqual(data['memory_samples'], 1)
        finally:
            agent.stop()

    def test_dashboard_disabled_by_default(self):
        # 重置Config单例
        from src.core.config import Config
        Config._instance = None
        Config._config = {}

        # 创建没有dashboard配置的文件
        with open('config_no_dashboard.yaml', 'w') as f:
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
  interval_hours: 6
  min_samples: 100
memory:
  max_short_term_samples: 100
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
""")

        agent = AgentLoop('config_no_dashboard.yaml')
        self.assertIsNone(agent.dashboard)
        agent.stop()

    def test_dashboard_history_accumulation(self):
        agent = AgentLoop('config.yaml')
        agent.dashboard.start()
        port = agent.dashboard._server.server_address[1]

        try:
            # 运行多个周期
            for _ in range(5):
                agent.run_cycle()

            time.sleep(0.3)

            req = urllib.request.Request(f'http://127.0.0.1:{port}/api/data')
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                self.assertEqual(data['cycles'], 5)
                self.assertEqual(len(data['history']), 5)
        finally:
            agent.stop()


if __name__ == '__main__':
    unittest.main()
