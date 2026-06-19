import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cli.main import AgentCLI, main


class TestAgentCLI(unittest.TestCase):
    def setUp(self):
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
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cmd_cycle(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            json = False
        args = Args()
        result = cli.cmd_cycle(args)
        self.assertEqual(result, 0)

    def test_cmd_cycle_json(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            json = True
        args = Args()
        result = cli.cmd_cycle(args)
        self.assertEqual(result, 0)

    def test_cmd_status(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            json = False
        args = Args()
        result = cli.cmd_status(args)
        self.assertEqual(result, 0)

    def test_cmd_status_json(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            json = True
        args = Args()
        result = cli.cmd_status(args)
        self.assertEqual(result, 0)

    def test_cmd_config_show(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            action = 'show'
        args = Args()
        result = cli.cmd_config(args)
        self.assertEqual(result, 0)

    def test_cmd_config_validate(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            action = 'validate'
        args = Args()
        result = cli.cmd_config(args)
        self.assertEqual(result, 0)

    def test_cmd_config_init(self):
        cli = AgentCLI()
        class Args:
            config = 'new_config.yaml'
            action = 'init'
            force = False
        args = Args()
        result = cli.cmd_config(args)
        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists('new_config.yaml'))

    def test_cmd_config_init_force(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            action = 'init'
            force = True
        args = Args()
        result = cli.cmd_config(args)
        self.assertEqual(result, 0)

    def test_cmd_config_init_exists_no_force(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            action = 'init'
            force = False
        args = Args()
        result = cli.cmd_config(args)
        self.assertEqual(result, 1)

    def test_cmd_cycle_missing_config(self):
        cli = AgentCLI()
        class Args:
            config = 'missing.yaml'
            json = False
        args = Args()
        result = cli.cmd_cycle(args)
        self.assertEqual(result, 1)

    def test_cmd_start_short_run(self):
        cli = AgentCLI()
        class Args:
            config = 'config.yaml'
            cycles = 2
            interval = 0.01
        args = Args()
        result = cli.cmd_start(args)
        self.assertEqual(result, 0)


class TestMainEntry(unittest.TestCase):
    def test_main_no_args(self):
        # 测试无参数时打印帮助（需要清空sys.argv避免pytest参数干扰）
        import sys
        old_argv = sys.argv
        sys.argv = ['agent']
        try:
            result = main()
            self.assertEqual(result, 0)
        finally:
            sys.argv = old_argv


if __name__ == '__main__':
    unittest.main()
