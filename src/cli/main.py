#!/usr/bin/env python3
"""CLI命令行入口：提供启动、停止、状态查询、配置管理、健康检查等功能"""

import abc
import argparse
import importlib.metadata
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, List, Optional, Tuple

# 确保src在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.constants import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


def _get_version() -> str:
    """获取项目版本号。

    优先从已安装的包元数据读取，回退到硬编码版本。
    """
    try:
        return importlib.metadata.version('self-evolving-agent')
    except importlib.metadata.PackageNotFoundError:
        return '1.0.0'


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
        logger.info("Received signal %s, shutting down...", signum)
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
            logger.error("Config file not found: %s", config_path)
            return False
        return True

    def _run_agent(self, config_path: str, args: argparse.Namespace) -> int:
        logger.info("Starting Self-Evolving Agent with config: %s", config_path)
        self.agent = AgentLoop(config_path)
        pid_path = os.path.join(self.agent.cfg.get('paths', {}).get('data_dir', 'data'), 'agent.pid')
        self._write_pid_file(pid_path)
        logger.info("Agent initialized. Running %s cycles...", args.cycles)
        try:
            self.agent.run(max_cycles=args.cycles, sleep_interval=args.interval)
        except Exception as e:
            logger.error("Agent error: %s", e)
            return 1
        finally:
            self._remove_pid_file(pid_path)
            if self.agent:
                self.agent.stop()
        logger.info("Agent stopped.")
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
            logger.info("Agent is not running (no PID file found).")
            return 0
        try:
            with open(pid_path, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to Agent process %s.", pid)
        except ProcessLookupError:
            logger.info("Agent process not found, cleaning up PID file.")
            self._remove_pid_file(pid_path)
        except Exception as e:
            logger.error("Error stopping agent: %s", e)
            return 1
        return 0

    def cmd_status(self, args: argparse.Namespace) -> int:
        """查询Agent状态"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1
        stats = self._get_lightweight_status(config_path)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            self._print_status(stats)
        return 0

    def _get_lightweight_status(self, config_path: str) -> Dict[str, Any]:
        """轻量级状态查询：不实例化 AgentLoop，直接读取配置/PID/DB。

        :param config_path: 配置文件路径。
        :type config_path: str
        :return: 状态字典。
        :rtype: Dict[str, Any]
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

        paths = cfg.get('paths', {})
        data_dir = paths.get('data_dir', 'data')
        log_dir = paths.get('log_dir', 'logs')

        # 检查运行状态（PID 文件）
        pid_path = os.path.join(data_dir, 'agent.pid')
        running = False
        pid = None
        if os.path.exists(pid_path):
            try:
                with open(pid_path, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # 检查进程是否存在
                running = True
            except (ValueError, ProcessLookupError, PermissionError):
                running = False

        # 轻量读取记忆样本数（直接查 SQLite，不实例化 AgentLoop）
        memory_samples = self._count_memory_samples(data_dir)

        # 读取最近日志的 MAE 与成功率（从状态向量持久化文件，若存在）
        recent_mae, recent_success_rate = self._read_recent_metrics(data_dir)

        return {
            'running': running,
            'pid': pid,
            'cycle_count': self._read_cycle_count(data_dir),
            'memory_samples': memory_samples,
            'recent_mae': recent_mae,
            'recent_success_rate': recent_success_rate,
            'config_path': config_path,
            'log_dir': log_dir,
        }

    def _count_memory_samples(self, data_dir: str) -> int:
        """直接查询 SQLite 获取样本数，不实例化 ShortTermMemory。"""
        db_path = os.path.join(data_dir, 'memory.db')
        if not os.path.exists(db_path):
            return 0
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute('SELECT COUNT(*) FROM samples')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except sqlite3.Error:
            return 0

    def _read_cycle_count(self, data_dir: str) -> int:
        """从快照索引读取周期数（轻量，不实例化 AgentLoop）。"""
        index_path = os.path.join('snapshots', 'index.json')
        if not os.path.exists(index_path):
            return 0
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
            if isinstance(index, list) and index:
                return max((s.get('cycle', 0) for s in index), default=0)
            if isinstance(index, dict):
                return index.get('cycle_count', 0)
        except (json.JSONDecodeError, OSError):
            pass
        return 0

    def _read_recent_metrics(self, data_dir: str) -> tuple:
        """读取最近的 MAE 与成功率（从指标文件，若存在）。"""
        metrics_path = os.path.join(data_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            return 0.0, 0.0
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics.get('recent_mae', 0.0), metrics.get('recent_success_rate', 0.0)
        except (json.JSONDecodeError, OSError):
            return 0.0, 0.0

    def _get_agent_stats(self, config_path: str) -> Dict[str, Any]:
        """完整状态查询（实例化 AgentLoop，保留以兼容旧调用）。"""
        agent = AgentLoop(config_path)
        stats = agent.get_stats()
        agent.stop()
        return stats

    def _print_status(self, stats: Dict[str, Any]) -> None:
        print("=" * 40)
        print("Agent Status")
        print("=" * 40)
        running = stats.get('running', False)
        print(f"Running:          {'YES' if running else 'NO'}")
        if stats.get('pid'):
            print(f"PID:              {stats['pid']}")
        print(f"Cycles Completed: {stats.get('cycle_count', 0)}")
        print(f"Memory Samples:   {stats.get('memory_samples', 0)}")
        mae = stats.get('recent_mae', 0.0)
        sr = stats.get('recent_success_rate', 0.0)
        print(f"Recent MAE:       {mae:.4f}" if isinstance(mae, (int, float)) else f"Recent MAE:       {mae}")
        if isinstance(sr, (int, float)):
            print(f"Success Rate:     {sr:.2%}")
        else:
            print(f"Success Rate:     {sr}")
        print(f"Config:           {stats.get('config_path', '-')}")
        print(f"Log Dir:          {stats.get('log_dir', '-')}")
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
            logger.error("Config file not found: %s", config_path)
            return 1
        with open(config_path, 'r') as f:
            print(f.read())
        return 0

    def _config_validate(self, config_path: str) -> int:
        try:
            agent = AgentLoop(config_path)
            agent.stop()
            logger.info("Configuration is valid.")
        except Exception as e:
            logger.error("Configuration error: %s", e)
            return 1
        return 0

    def _config_init(self, config_path: str, force: bool) -> int:
        if os.path.exists(config_path) and not force:
            logger.error("Config already exists: %s", config_path)
            logger.error("Use --force to overwrite.")
            return 1
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(self._default_config())
        logger.info("Default config written to: %s", config_path)
        print(f"Config created: {config_path}")
        print()
        print("Next steps:")
        print(f"  1. Edit {config_path} to adjust settings")
        print("  2. Run 'self-evolving-agent config validate' to check")
        print("  3. Run 'self-evolving-agent doctor' to verify environment")
        print("  4. Run 'self-evolving-agent start' to launch")
        return 0

    def _default_config(self) -> str:
        """生成带注释的默认配置模板，便于用户理解各字段含义。"""
        return """# Self-Evolving Agent 配置文件
# 生成命令: self-evolving-agent config init
# 完整文档: https://github.com/17683995446/self_evolving_agent

# === 进化目标开关 ===
# 控制系统自主进化的方向
goals:
  precision_evolution: true    # 预测精度进化（降低 MAE）
  coverage_evolution: true     # 行为覆盖进化（探索新任务）
  efficiency_evolution: true   # 能效进化（降低资源消耗）

# === 驱力权重 ===
# 三大驱力的权重，和为 1.0，决定任务优先级
base_weights:
  safe: 0.4    # 安全驱力：避免危险行为
  gain: 0.4    # 收益驱力：追求预测精度提升
  eff: 0.2     # 能效驱力：降低资源消耗

# === 资源限制 ===
# Agent 运行时的资源上限，超出触发紧急模式
resource_limits:
  max_cpu_percent: 20.0        # CPU 使用率上限（%）
  max_mem_percent: 20.0        # 内存使用率上限（%）
  max_db_size_mb: 500          # 记忆数据库大小上限（MB）
  max_snapshot_count: 10       # 模型快照保留数量

# === 探索策略 ===
# 控制任务生成的探索行为比例
exploration:
  base_ratio: 0.3              # 基础探索比例
  stable_threshold: 0.05       # 稳定性判定阈值
  emergency_ratio: 0.1         # 紧急模式探索比例
  novel_ratio: 0.4             # 新颖任务比例

# === 预测模型参数 ===
models:
  smoothing_alpha: 0.3         # 指数平滑系数（0~1，越大越敏感）
  smoothing_alpha_min: 0.05    # 平滑系数下限
  smoothing_alpha_max: 0.8     # 平滑系数上限
  arima_order: [1, 1, 1]       # ARIMA 阶数 [p, d, q]
  prediction_horizon: 1        # 预测步长

# === 安全约束 ===
safety:
  max_consecutive_failures: 5          # 连续失败上限，超出冻结任务
  forbidden_paths:                     # 禁止访问的路径
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
  max_model_restructure_iterations: 3  # 模型重构最大迭代次数

# === 离线复盘 ===
# 定期复盘历史数据，优化模型
review:
  interval_hours: 6            # 复盘间隔（小时）
  min_samples: 100             # 触发复盘的最小样本数
  idle_cpu_threshold: 10.0     # CPU 空闲阈值（%）
  idle_duration_seconds: 300   # 空闲持续时长（秒）

# === 记忆管理 ===
memory:
  max_short_term_samples: 10000    # 短期记忆最大样本数
  value_retention_ratio: 0.3       # 高价值样本保留比例
  history_window: 100              # 历史窗口大小

# === 日志配置 ===
logging:
  level: INFO                                      # 日志级别：DEBUG/INFO/WARNING/ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log                             # 日志文件路径
  max_bytes: 10485760                              # 单文件最大字节数（10MB）
  backup_count: 5                                  # 保留备份文件数

# === 数据路径 ===
paths:
  data_dir: data        # 数据目录（记忆数据库等）
  snapshot_dir: snapshots  # 快照目录（模型快照）
  log_dir: logs         # 日志目录
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

    def cmd_doctor(self, args: argparse.Namespace) -> int:
        """健康检查：验证运行环境、依赖、配置、路径权限。

        :param args: 命令行参数。
        :type args: argparse.Namespace
        :return: 全部通过返回0，存在问题返回1。
        :rtype: int
        """
        print(f"Self-Evolving Agent v{_get_version()} - Health Check")
        print("=" * 50)

        checks: List[Tuple[str, bool, str]] = []
        checks.extend(self._check_python_version())
        checks.extend(self._check_dependencies())
        checks.extend(self._check_config(args))
        checks.extend(self._check_paths(args))

        passed = sum(1 for _, ok, _ in checks if ok)
        total = len(checks)
        for name, ok, msg in checks:
            status = "[OK]  " if ok else "[FAIL]"
            print(f"  {status} {name}: {msg}")

        print("=" * 50)
        print(f"Result: {passed}/{total} checks passed")
        if passed == total:
            print("All checks passed. System is ready.")
            return 0
        print("Some checks failed. Please review the issues above.")
        return 1

    def _check_python_version(self) -> List[Tuple[str, bool, str]]:
        """检查 Python 版本。"""
        major, minor = sys.version_info[:2]
        ok = (major, minor) >= (3, 10)
        return [('Python Version', ok, f"{major}.{minor}")]

    def _check_dependencies(self) -> List[Tuple[str, bool, str]]:
        """检查关键依赖包是否可导入。"""
        deps = [
            ('psutil', 'psutil'),
            ('numpy', 'numpy'),
            ('statsmodels', 'statsmodels'),
            ('yaml (PyYAML)', 'yaml'),
            ('watchdog', 'watchdog'),
        ]
        results: List[Tuple[str, bool, str]] = []
        for label, module_name in deps:
            try:
                mod = __import__(module_name)
                version = getattr(mod, '__version__', 'unknown')
                results.append((label, True, version))
            except ImportError as e:
                results.append((label, False, f"not installed: {e}"))
        return results

    def _check_config(self, args: argparse.Namespace) -> List[Tuple[str, bool, str]]:
        """检查配置文件是否存在且可加载。"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not os.path.exists(config_path):
            return [('Config File', False, f"not found: {config_path}")]
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return [('Config File', True, config_path)]
        except Exception as e:
            return [('Config File', False, f"parse error: {e}")]

    def _check_paths(self, args: argparse.Namespace) -> List[Tuple[str, bool, str]]:
        """检查数据路径是否可创建/可写。"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        required_dirs = ['data', 'logs', 'snapshots']
        try:
            import yaml
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                paths = cfg.get('paths', {})
                required_dirs = [
                    paths.get('data_dir', 'data'),
                    paths.get('log_dir', 'logs'),
                    paths.get('snapshot_dir', 'snapshots'),
                ]
        except Exception:
            pass

        results: List[Tuple[str, bool, str]] = []
        for d in required_dirs:
            try:
                os.makedirs(d, exist_ok=True)
                test_file = os.path.join(d, '.doctor_write_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                results.append((f"Path: {d}", True, "writable"))
            except OSError as e:
                results.append((f"Path: {d}", False, f"not writable: {e}"))
        return results

    def cmd_logs(self, args: argparse.Namespace) -> int:
        """查看 Agent 日志（尾部 N 行）。

        :param args: 命令行参数，支持 --lines 指定行数。
        :type args: argparse.Namespace
        :return: 成功返回0，日志文件不存在返回1。
        :rtype: int
        """
        config_path = args.config or DEFAULT_CONFIG_PATH
        log_file = self._resolve_log_file(config_path)
        if log_file is None or not os.path.exists(log_file):
            logger.error("Log file not found: %s", log_file or '(none)')
            return 1
        lines = self._read_log_tail(log_file, args.lines)
        for line in lines:
            print(line, end='' if line.endswith('\n') else '\n')
        return 0

    def _resolve_log_file(self, config_path: str) -> Optional[str]:
        """从配置文件解析日志文件路径。"""
        try:
            import yaml
            if not os.path.exists(config_path):
                return None
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            log_dir = cfg.get('paths', {}).get('log_dir', 'logs')
            log_file = cfg.get('logging', {}).get('file', 'logs/agent.log').split('/')[-1]
            return os.path.join(log_dir, log_file)
        except Exception:
            return None

    def _read_log_tail(self, log_file: str, n: int) -> List[str]:
        """读取日志文件尾部 n 行。"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
            return all_lines[-n:] if n > 0 else all_lines
        except OSError as e:
            logger.error("Failed to read log file %s: %s", log_file, e)
            return []


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
  self-evolving-agent doctor                  # 健康检查
  self-evolving-agent config init             # 初始化配置
  self-evolving-agent start --cycles 10       # 启动运行10个周期
  self-evolving-agent status --json           # JSON格式查看状态
  self-evolving-agent cycle                   # 执行单次循环
        """
    )
    parser.add_argument('--version', '-V', action='version',
                        version=f'self-evolving-agent {_get_version()}')
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

    # doctor
    subparsers.add_parser('doctor', help='Run health check (deps, config, paths)')

    # logs
    logs_parser = subparsers.add_parser('logs', help='Show recent log lines')
    logs_parser.add_argument('--lines', '-n', type=int, default=50,
                             help='Number of lines to show (default: 50)')

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
        'doctor': cli.cmd_doctor,
        'logs': cli.cmd_logs,
    }

    handler = command_map.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
