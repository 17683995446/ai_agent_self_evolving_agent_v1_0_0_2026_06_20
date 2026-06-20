#!/usr/bin/env python3
"""CLI命令行入口：提供启动、停止、状态查询、配置管理等功能"""

import abc
import argparse
import sys
import os
import json
import signal
from typing import Optional, Dict, Any

# 确保src在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.constants import DEFAULT_CONFIG_PATH


class BaseAgentCLI(abc.ABC):
    """Agent命令行接口抽象基类。"""

    def __init__(self):
        self.agent: Optional[AgentLoop] = None
        self._shutdown_requested = False

    @abc.abstractmethod
    def cmd_start(self, args: argparse.Namespace) -> int:
        """启动Agent主循环。"""

    @abc.abstractmethod
    def cmd_stop(self, args: argparse.Namespace) -> int:
        """停止Agent主循环。"""

    @abc.abstractmethod
    def cmd_status(self, args: argparse.Namespace) -> int:
        """查询Agent状态。"""

    @abc.abstractmethod
    def cmd_config(self, args: argparse.Namespace) -> int:
        """配置管理。"""

    @abc.abstractmethod
    def cmd_cycle(self, args: argparse.Namespace) -> int:
        """执行单次循环。"""


class AgentCLI(BaseAgentCLI):
    """Agent命令行接口"""

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        print(f"\nReceived signal {signum}, shutting down...")
        self._shutdown_requested = True
        if self.agent:
            self.agent.stop()

    def _write_pid_file(self, pid_path: str) -> None:
        with open(pid_path, 'w') as f:
            f.write(str(os.getpid()))

    def _remove_pid_file(self, pid_path: str) -> None:
        if os.path.exists(pid_path):
            os.remove(pid_path)

    def cmd_start(self, args: argparse.Namespace) -> int:
        """启动Agent主循环"""
        self._setup_signal_handlers()
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1
        return self._run_agent(config_path, args)

    def _validate_config_path(self, config_path: str) -> bool:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            return False
        return True

    def _run_agent(self, config_path: str, args: argparse.Namespace) -> int:
        print(f"Starting Self-Evolving Agent with config: {config_path}")
        self.agent = AgentLoop(config_path)
        pid_path = os.path.join(self.agent.cfg.get('paths', {}).get('data_dir', 'data'), 'agent.pid')
        self._write_pid_file(pid_path)
        print(f"Agent initialized. Running {args.cycles} cycles...")
        try:
            self.agent.run(max_cycles=args.cycles, sleep_interval=args.interval)
        except Exception as e:
            print(f"Agent error: {e}")
            return 1
        finally:
            self._remove_pid_file(pid_path)
            if self.agent:
                self.agent.stop()
        print("Agent stopped.")
        return 0

    def cmd_stop(self, args: argparse.Namespace) -> int:
        """停止Agent主循环"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1
        agent = AgentLoop(config_path)
        pid_path = os.path.join(agent.cfg.get('paths', {}).get('data_dir', 'data'), 'agent.pid')
        agent.stop()
        if not os.path.exists(pid_path):
            print("Agent is not running (no PID file found).")
            return 0
        try:
            with open(pid_path, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to Agent process {pid}.")
        except ProcessLookupError:
            print("Agent process not found, cleaning up PID file.")
            self._remove_pid_file(pid_path)
        except Exception as e:
            print(f"Error stopping agent: {e}")
            return 1
        return 0

    def cmd_status(self, args: argparse.Namespace) -> int:
        """查询Agent状态"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1
        stats = self._get_agent_stats(config_path)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            self._print_status(stats)
        return 0

    def _get_agent_stats(self, config_path: str) -> Dict[str, Any]:
        agent = AgentLoop(config_path)
        stats = agent.get_stats()
        agent.stop()
        return stats

    def _print_status(self, stats: Dict[str, Any]) -> None:
        print("=" * 40)
        print("Agent Status")
        print("=" * 40)
        print(f"Cycles Completed: {stats['cycle_count']}")
        print(f"Memory Samples:   {stats['memory_samples']}")
        print(f"Recent MAE:       {stats['recent_mae']:.4f}")
        print(f"Success Rate:     {stats['recent_success_rate']:.2%}")
        print("-" * 40)
        print("Task Pool:")
        for key, value in stats['task_pool'].items():
            print(f"  {key}: {value}")
        print("=" * 40)

    def cmd_config(self, args: argparse.Namespace) -> int:
        """配置管理"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if args.action == 'show':
            return self._config_show(config_path)
        elif args.action == 'validate':
            return self._config_validate(config_path)
        elif args.action == 'init':
            return self._config_init(config_path, args.force)
        return 0

    def _config_show(self, config_path: str) -> int:
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return 1
        with open(config_path, 'r') as f:
            print(f.read())
        return 0

    def _config_validate(self, config_path: str) -> int:
        try:
            agent = AgentLoop(config_path)
            agent.stop()
            print("Configuration is valid.")
        except Exception as e:
            print(f"Configuration error: {e}")
            return 1
        return 0

    def _config_init(self, config_path: str, force: bool) -> int:
        if os.path.exists(config_path) and not force:
            print(f"Config already exists: {config_path}")
            print("Use --force to overwrite.")
            return 1
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(self._default_config())
        print(f"Default config written to: {config_path}")
        return 0

    def _default_config(self) -> str:
        return """goals:
  precision_evolution: true
  coverage_evolution: true
  efficiency_evolution: true

base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2

resource_limits:
  max_cpu_percent: 20.0
  max_mem_percent: 20.0
  max_db_size_mb: 500
  max_snapshot_count: 10

exploration:
  base_ratio: 0.3
  stable_threshold: 0.05
  emergency_ratio: 0.1
  novel_ratio: 0.4

models:
  smoothing_alpha: 0.3
  smoothing_alpha_min: 0.05
  smoothing_alpha_max: 0.8
  arima_order: [1, 1, 1]
  prediction_horizon: 1

safety:
  max_consecutive_failures: 5
  forbidden_paths:
    - /boot
    - /etc
    - /proc
    - /sys
    - /dev
    - /bin
    - /sbin
    - /lib
    - /lib64
    - /usr/bin
    - /usr/sbin
  max_model_restructure_iterations: 3

review:
  interval_hours: 6
  min_samples: 100
  idle_cpu_threshold: 10.0
  idle_duration_seconds: 300

memory:
  max_short_term_samples: 10000
  value_retention_ratio: 0.3
  history_window: 100

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log
  max_bytes: 10485760
  backup_count: 5

paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
"""

    def cmd_cycle(self, args: argparse.Namespace) -> int:
        """执行单次循环（用于调试和测试）"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1
        result = self._run_single_cycle(config_path)
        if args.json:
            print(json.dumps(self._simplify_result(result), indent=2))
        else:
            self._print_cycle_result(result)
        return 0

    def _run_single_cycle(self, config_path: str) -> Dict[str, Any]:
        agent = AgentLoop(config_path)
        result = agent.run_cycle()
        agent.stop()
        return result

    def _simplify_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'cycle': result['cycle'],
            'errors': result['errors'],
            'reward': result['reward'],
            'execution': result['execution'],
            'safety': result['safety'],
            'duration_seconds': result['duration_seconds'],
        }

    def _print_cycle_result(self, result: Dict[str, Any]) -> None:
        print(f"Cycle {result['cycle']} completed in {result['duration_seconds']:.4f}s")
        print(f"Errors: {result['errors']}")
        print(f"Reward: {result['reward']['total_reward']:.4f}")
        print(f"Execution: {result['execution']['status']}")
        print(f"Safety: emergency={result['safety']['emergency_mode']}")


def _resolve_config_path(args_config: Optional[str]) -> str:
    if args_config:
        return args_config
    env = os.environ.get('AGENT_ENV', 'prod').lower()
    env_map = {
        'dev': 'config/config.dev.yaml',
        'test': 'config/config.test.yaml',
        'prod': 'config/config.prod.yaml',
    }
    return env_map.get(env, 'config/config.prod.yaml')


def main() -> int:
    cli = AgentCLI()
    parser = argparse.ArgumentParser(
        description="Self-Evolving Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.main start --cycles 10
  python -m src.cli.main status
  python -m src.cli.main config init
  python -m src.cli.main cycle --json
        """
    )
    parser.add_argument('--config', '-c', help='Path to config file')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # start
    start_parser = subparsers.add_parser('start', help='Start the agent')
    start_parser.add_argument('--cycles', '-n', type=int, default=None,
                              help='Number of cycles to run (default: infinite)')
    start_parser.add_argument('--interval', '-i', type=float, default=1.0,
                              help='Sleep interval between cycles (seconds)')

    # status
    subparsers.add_parser('status', help='Show agent status')

    # config
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('action', choices=['show', 'validate', 'init'])
    config_parser.add_argument('--force', action='store_true', help='Force overwrite')

    # stop
    subparsers.add_parser('stop', help='Stop the agent')

    # cycle
    subparsers.add_parser('cycle', help='Run a single cycle')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    args.config = _resolve_config_path(args.config)

    command_map = {
        'start': cli.cmd_start,
        'stop': cli.cmd_stop,
        'status': cli.cmd_status,
        'config': cli.cmd_config,
        'cycle': cli.cmd_cycle,
    }

    handler = command_map.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
