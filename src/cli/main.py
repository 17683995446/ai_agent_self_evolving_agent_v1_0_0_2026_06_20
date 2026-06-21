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
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

# 确保src在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.constants import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


class ColorOutput:
    """终端彩色输出工具，自动检测 TTY 并尊重 NO_COLOR 环境变量。

    遵循 https://no-color.org/ 规范：设置 NO_COLOR 环境变量时禁用颜色。
    输出非 TTY（如管道/重定向）时自动禁用颜色，避免 ANSI 转义码污染日志。
    """

    _RESET = '\033[0m'
    _COLORS = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'cyan': '\033[36m',
        'gray': '\033[90m',
        'bold': '\033[1m',
    }

    def __init__(self):
        self._enabled = self._detect_tty()

    def _detect_tty(self) -> bool:
        """检测是否应启用颜色输出。"""
        if os.environ.get('NO_COLOR') is not None:
            return False
        return sys.stdout.isatty()

    def disable(self) -> None:
        """禁用颜色输出（由 --no-color 选项调用）。"""
        self._enabled = False

    def colorize(self, text: str, color: str) -> str:
        """为文本添加颜色，未启用时返回原文。"""
        if not self._enabled or color not in self._COLORS:
            return text
        return f"{self._COLORS[color]}{text}{self._RESET}"

    def ok(self, text: str) -> str:
        """绿色：成功状态。"""
        return self.colorize(text, 'green')

    def fail(self, text: str) -> str:
        """红色：失败状态。"""
        return self.colorize(text, 'red')

    def warn(self, text: str) -> str:
        """黄色：警告状态。"""
        return self.colorize(text, 'yellow')

    def info(self, text: str) -> str:
        """青色：信息标题。"""
        return self.colorize(text, 'cyan')

    def bold(self, text: str) -> str:
        """粗体。"""
        return self.colorize(text, 'bold')

    def is_enabled(self) -> bool:
        """返回颜色是否启用。"""
        return self._enabled


# 全局颜色输出实例
_color = ColorOutput()

# 统一分隔符宽度
SEPARATOR_WIDTH = 50


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
            print()
            print("Hint: Generate a default config with:")
            print(f"  self-evolving-agent -c {config_path} config init")
            print("Or run 'self-evolving-agent quickstart' for a guided setup.")
            return False
        return True

    def _run_agent(self, config_path: str, args: argparse.Namespace) -> int:
        logger.info("Starting Self-Evolving Agent with config: %s", config_path)
        try:
            self.agent = AgentLoop(config_path)
        except FileNotFoundError as e:
            logger.error("Config file error: %s", e)
            return 1
        except Exception as e:
            logger.error("Agent initialization failed: %s", e)
            print()
            print("Hint: Run 'self-evolving-agent config validate' to check config.")
            return 1
        pid_path = os.path.join(self.agent.cfg.get('paths', {}).get('data_dir', 'data'), 'agent.pid')
        self._write_pid_file(pid_path)
        max_cycles = args.cycles
        if max_cycles is not None:
            print(f"Agent started: running {max_cycles} cycles "
                  f"(interval={args.interval}s)")
        else:
            print("Agent started: running indefinitely (Ctrl+C to stop)")
        print()
        self._progress_start = time.time()
        self._progress_max = max_cycles
        try:
            self.agent.run(max_cycles=max_cycles, sleep_interval=args.interval,
                           progress_callback=self._print_progress)
        except KeyboardInterrupt:
            logger.info("Agent stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error("Agent runtime error: %s", e)
            print()
            print("Hint: Check logs with 'self-evolving-agent logs --lines 50'")
            return 1
        finally:
            self._remove_pid_file(pid_path)
            if self.agent:
                self.agent.stop()
        print()
        print("Agent stopped.")
        return 0

    def _print_progress(self, result: Dict[str, Any], cycle_count: int,
                        max_cycles: Optional[int]) -> None:
        """每周期结束后打印进度信息。

        :param result: 周期结果字典。
        :param cycle_count: 已完成的周期数。
        :param max_cycles: 最大周期数（None 表示无限）。
        """
        elapsed = time.time() - self._progress_start
        reward = result.get('reward', {}).get('total_reward', 0.0)
        mae = 0.0
        errors = result.get('errors', {})
        if isinstance(errors, dict) and errors:
            mae = sum(abs(v) for v in errors.values()) / len(errors)

        if max_cycles is not None and max_cycles > 0:
            progress = cycle_count / max_cycles
            bar_width = 20
            filled = int(bar_width * progress)
            bar = '#' * filled + '-' * (bar_width - filled)
            pct = progress * 100
            eta = (elapsed / cycle_count * (max_cycles - cycle_count)) if cycle_count > 0 else 0
            print(f"\r  [{bar}] {pct:5.1f}%  "
                  f"cycle {cycle_count}/{max_cycles}  "
                  f"reward={reward:.4f}  mae={mae:.4f}  "
                  f"elapsed={elapsed:.1f}s  eta={eta:.1f}s", end='', flush=True)
        else:
            print(f"\r  cycle {cycle_count}  "
                  f"reward={reward:.4f}  mae={mae:.4f}  "
                  f"elapsed={elapsed:.1f}s", end='', flush=True)

    def cmd_stop(self, args: argparse.Namespace) -> int:
        """停止Agent主循环"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1
        # 轻量读取配置获取 PID 路径（不实例化 AgentLoop）
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            data_dir = cfg.get('paths', {}).get('data_dir', 'data')
        except Exception as e:
            logger.warning("Failed to read config for data_dir, using default 'data': %s", e)
            data_dir = 'data'
        pid_path = os.path.join(data_dir, 'agent.pid')
        if not os.path.exists(pid_path):
            print("Agent is not running (no PID file found).")
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
            except (ValueError, ProcessLookupError, PermissionError) as e:
                logger.debug("PID file read or process check failed: %s", e)
                running = False

        # 轻量读取记忆样本数（直接查 SQLite，不实例化 AgentLoop）
        memory_samples = self._count_memory_samples(data_dir)

        # 读取最近指标（从 metrics.json，由 AgentLoop 每周期写入）
        metrics = self._read_metrics(data_dir)
        recent_mae = metrics.get('recent_mae', 0.0)
        recent_success_rate = metrics.get('recent_success_rate', 0.0)
        cycle_count = metrics.get('cycle_count') or self._read_cycle_count(data_dir)

        return {
            'running': running,
            'pid': pid,
            'cycle_count': cycle_count,
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
        except sqlite3.Error as e:
            logger.debug("Failed to count memory samples from %s: %s", db_path, e)
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
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to read cycle count from %s: %s", index_path, e)
        return 0

    def _read_metrics(self, data_dir: str) -> Dict[str, Any]:
        """读取 metrics.json（由 AgentLoop 每周期写入）。

        :param data_dir: 数据目录。
        :type data_dir: str
        :return: 指标字典，文件不存在时返回空字典。
        :rtype: Dict[str, Any]
        """
        metrics_path = os.path.join(data_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            return {}
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to read metrics from %s: %s", metrics_path, e)
            return {}

    def _get_agent_stats(self, config_path: str) -> Dict[str, Any]:
        """完整状态查询（实例化 AgentLoop，保留以兼容旧调用）。"""
        agent = AgentLoop(config_path)
        stats = agent.get_stats()
        agent.stop()
        return stats

    def _print_status(self, stats: Dict[str, Any]) -> None:
        print("=" * SEPARATOR_WIDTH)
        print(_color.bold("Agent Status"))
        print("=" * SEPARATOR_WIDTH)
        running = stats.get('running', False)
        running_text = _color.ok('YES') if running else _color.warn('NO')
        print(f"Running:          {running_text}")
        if stats.get('pid'):
            print(f"PID:              {stats['pid']}")
        print(f"Cycles Completed: {stats.get('cycle_count', 0)}")
        print(f"Memory Samples:   {stats.get('memory_samples', 0)}")
        mae = stats.get('recent_mae', 0.0)
        sr = stats.get('recent_success_rate', 0.0)
        if isinstance(mae, (int, float)):
            mae_text = f"{mae:.4f}"
            if mae < 0.05:
                mae_text = _color.ok(mae_text)
            elif mae < 0.15:
                mae_text = _color.warn(mae_text)
            else:
                mae_text = _color.fail(mae_text)
            print(f"Recent MAE:       {mae_text}")
        else:
            print(f"Recent MAE:       {mae}")
        if isinstance(sr, (int, float)):
            sr_text = f"{sr:.2%}"
            if sr > 0.7:
                sr_text = _color.ok(sr_text)
            elif sr > 0.4:
                sr_text = _color.warn(sr_text)
            else:
                sr_text = _color.fail(sr_text)
            print(f"Success Rate:     {sr_text}")
        else:
            print(f"Success Rate:     {sr}")
        print(f"Config:           {stats.get('config_path', '-')}")
        print(f"Log Dir:          {stats.get('log_dir', '-')}")
        print("=" * SEPARATOR_WIDTH)

    def cmd_config(self, args: argparse.Namespace) -> int:
        """配置管理"""
        config_path = args.config or DEFAULT_CONFIG_PATH
        if args.action == 'show':
            return self._config_show(config_path)
        elif args.action == 'validate':
            return self._config_validate(config_path, args)
        elif args.action == 'init':
            return self._config_init(config_path, args.force)
        elif args.action == 'wizard':
            return self.cmd_config_wizard(args)
        return 0

    def _config_show(self, config_path: str) -> int:
        if not os.path.exists(config_path):
            logger.error("Config file not found: %s", config_path)
            return 1
        with open(config_path, 'r') as f:
            print(f.read())
        return 0

    def _config_validate(self, config_path: str,
                         args: Optional[argparse.Namespace] = None) -> int:
        """校验配置文件的字段合法性，输出友好的错误提示。

        :param config_path: 配置文件路径。
        :type config_path: str
        :param args: 命令行参数（用于检测 --json），可选。
        :type args: Optional[argparse.Namespace]
        :return: 全部校验通过返回0，存在错误返回1。
        :rtype: int
        """
        errors = self._validate_config_fields(config_path)
        json_output = getattr(args, 'json', False) if args else False

        if json_output:
            result = {
                'config_path': config_path,
                'valid': len(errors) == 0,
                'error_count': len(errors),
                'errors': errors,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if not errors else 1

        if not errors:
            print(_color.ok(f"Configuration is valid: {config_path}"))
            return 0
        print(_color.fail(f"Configuration validation failed ({len(errors)} errors):"))
        for err in errors:
            print(f"  - {_color.fail(err)}")
        print()
        print(_color.info("Fix hints:"))
        print("  - Run 'self-evolving-agent config init --force' to regenerate defaults")
        print("  - Check field types: weights must be numbers, paths must be strings")
        return 1

    def _validate_config_fields(self, config_path: str) -> List[str]:
        """校验配置字段，返回错误信息列表。

        :param config_path: 配置文件路径。
        :type config_path: str
        :return: 错误信息列表，空列表表示全部通过。
        :rtype: List[str]
        """
        errors: List[str] = []
        if not os.path.exists(config_path):
            return [f"Config file not found: {config_path}"]

        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return [f"YAML parse error: {e}"]
        except OSError as e:
            return [f"Cannot read file: {e}"]

        if cfg is None or not isinstance(cfg, dict):
            return ["Config file is empty or not a YAML mapping"]

        errors.extend(self._validate_base_weights(cfg))
        errors.extend(self._validate_resource_limits(cfg))
        errors.extend(self._validate_safety(cfg))
        errors.extend(self._validate_paths(cfg))
        errors.extend(self._validate_logging(cfg))
        return errors

    def _validate_base_weights(self, cfg: Dict[str, Any]) -> List[str]:
        """校验驱力权重：必须为数字且和约为1.0。"""
        errors: List[str] = []
        bw = cfg.get('base_weights')
        if bw is None:
            return ["Missing 'base_weights' section"]
        if not isinstance(bw, dict):
            return ["'base_weights' must be a mapping"]
        for key in ('safe', 'gain', 'eff'):
            val = bw.get(key)
            if val is None:
                errors.append(f"base_weights.{key} is missing")
            elif not isinstance(val, (int, float)):
                errors.append(f"base_weights.{key} must be a number, got {type(val).__name__}")
        if not errors:
            total = bw.get('safe', 0) + bw.get('gain', 0) + bw.get('eff', 0)
            if abs(total - 1.0) > 0.01:
                errors.append(f"base_weights sum should be ~1.0, got {total:.4f}")
        return errors

    def _validate_resource_limits(self, cfg: Dict[str, Any]) -> List[str]:
        """校验资源限制：百分比必须在 0~100。"""
        errors: List[str] = []
        rl = cfg.get('resource_limits', {})
        if not isinstance(rl, dict):
            return ["'resource_limits' must be a mapping"]
        for key in ('max_cpu_percent', 'max_mem_percent'):
            val = rl.get(key)
            if val is not None:
                if not isinstance(val, (int, float)):
                    errors.append(f"resource_limits.{key} must be a number")
                elif val < 0 or val > 100:
                    errors.append(f"resource_limits.{key} must be 0~100, got {val}")
        return errors

    def _validate_safety(self, cfg: Dict[str, Any]) -> List[str]:
        """校验安全配置。"""
        errors: List[str] = []
        safety = cfg.get('safety', {})
        if not isinstance(safety, dict):
            return ["'safety' must be a mapping"]
        mcf = safety.get('max_consecutive_failures')
        if mcf is not None and (not isinstance(mcf, int) or mcf < 0):
            errors.append("safety.max_consecutive_failures must be a non-negative integer")
        fp = safety.get('forbidden_paths')
        if fp is not None and not isinstance(fp, list):
            errors.append("safety.forbidden_paths must be a list")
        return errors

    def _validate_paths(self, cfg: Dict[str, Any]) -> List[str]:
        """校验路径配置：必须为字符串。"""
        errors: List[str] = []
        paths = cfg.get('paths', {})
        if not isinstance(paths, dict):
            return ["'paths' must be a mapping"]
        for key in ('data_dir', 'snapshot_dir', 'log_dir'):
            val = paths.get(key)
            if val is not None and not isinstance(val, str):
                errors.append(f"paths.{key} must be a string, got {type(val).__name__}")
        return errors

    def _validate_logging(self, cfg: Dict[str, Any]) -> List[str]:
        """校验日志配置。"""
        errors: List[str] = []
        log_cfg = cfg.get('logging', {})
        if not isinstance(log_cfg, dict):
            return ["'logging' must be a mapping"]
        level = log_cfg.get('level')
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if level is not None:
            if not isinstance(level, str) or level.upper() not in valid_levels:
                errors.append(f"logging.level must be one of {valid_levels}, got {level}")
        # 校验 format_type（可选，text/json）
        format_type = log_cfg.get('format_type')
        if format_type is not None:
            if not isinstance(format_type, str) or format_type.lower() not in {'text', 'json'}:
                errors.append(f"logging.format_type must be 'text' or 'json', got {format_type}")
        # 校验 loggers（可选，模块名→级别映射）
        loggers_cfg = log_cfg.get('loggers')
        if loggers_cfg is not None:
            if not isinstance(loggers_cfg, dict):
                errors.append("logging.loggers must be a mapping of module name to level")
            else:
                for mod_name, mod_level in loggers_cfg.items():
                    if not isinstance(mod_level, str) or mod_level.upper() not in valid_levels:
                        errors.append(
                            f"logging.loggers['{mod_name}'] must be one of {valid_levels}, "
                            f"got {mod_level}"
                        )
        return errors

    def cmd_config_wizard(self, args: argparse.Namespace) -> int:
        """交互式配置向导：通过问答生成定制化配置。

        :param args: 命令行参数。
        :type args: argparse.Namespace
        :return: 成功返回0，用户取消返回1。
        :rtype: int
        """
        config_path = args.config or DEFAULT_CONFIG_PATH
        print("=" * SEPARATOR_WIDTH)
        print("  Self-Evolving Agent - 配置向导")
        print("  交互式配置向导，按 Ctrl+C 取消")
        print("=" * SEPARATOR_WIDTH)
        print()

        try:
            cfg = self._run_wizard_questions()
        except KeyboardInterrupt:
            print("\n向导已取消。")
            return 1

        if os.path.exists(config_path):
            print(f"\n配置已存在: {config_path}")
            confirm = input("是否覆盖? (y/N): ").strip().lower()
            if confirm != 'y':
                print("已取消。")
                return 0

        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(self._default_config())
            f.write('\n# === 向导自定义设置 ===\n')
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"\n配置已创建: {config_path}")
        print("后续步骤:")
        print(f"  1. 运行 'self-evolving-agent -c {config_path} config validate'")
        print(f"  2. 运行 'self-evolving-agent -c {config_path} doctor'")
        print(f"  3. 运行 'self-evolving-agent -c {config_path} start'")
        return 0

    def _run_wizard_questions(self) -> Dict[str, Any]:
        """执行向导问答，返回用户定制的配置字典。"""
        print("选择运行模式（影响资源限制和探索行为）:")
        print("  1. conservative（保守: CPU<=10%, 内存<=10%, 低探索）")
        print("  2. balanced     （均衡: CPU<=20%, 内存<=20%, 中探索）[默认]")
        print("  3. aggressive   （激进: CPU<=50%, 内存<=50%, 高探索）")
        profile = input("请选择 [1/2/3]: ").strip() or '2'

        profiles = {
            '1': {'cpu': 10.0, 'mem': 10.0, 'explore': 0.2, 'name': 'conservative'},
            '2': {'cpu': 20.0, 'mem': 20.0, 'explore': 0.3, 'name': 'balanced'},
            '3': {'cpu': 50.0, 'mem': 50.0, 'explore': 0.5, 'name': 'aggressive'},
        }
        p = profiles.get(profile, profiles['2'])
        print(f"  -> 已选择: {p['name']}")
        print()

        log_level = input("日志级别 [DEBUG/INFO/WARNING/ERROR]（默认 INFO）: ").strip().upper() or 'INFO'
        print(f"  -> 日志级别: {log_level}")
        print()

        data_dir = input("数据目录（默认 'data'）: ").strip() or 'data'
        print(f"  -> 数据目录: {data_dir}")
        print()

        return {
            'resource_limits': {
                'max_cpu_percent': p['cpu'],
                'max_mem_percent': p['mem'],
                'max_db_size_mb': 500,
                'max_snapshot_count': 10,
            },
            'exploration': {
                'base_ratio': p['explore'],
                'stable_threshold': 0.05,
                'emergency_ratio': 0.1,
                'novel_ratio': 0.4,
            },
            'logging': {
                'level': log_level,
            },
            'paths': {
                'data_dir': data_dir,
                'snapshot_dir': 'snapshots',
                'log_dir': 'logs',
            },
        }

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
        verbose = getattr(args, 'verbose', False)
        result = self._run_single_cycle(config_path)
        if args.json:
            print(json.dumps(self._simplify_result(result), indent=2))
        else:
            self._print_cycle_result(result, verbose)
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

    def _print_cycle_result(self, result: Dict[str, Any], verbose: bool = False) -> None:
        print(f"Cycle {_color.bold(str(result['cycle']))} "
              f"completed in {result['duration_seconds']:.4f}s")
        print(f"Errors: {result['errors']}")
        reward_val = result['reward']['total_reward']
        reward_text = f"{reward_val:.4f}"
        if reward_val > 0.5:
            reward_text = _color.ok(reward_text)
        elif reward_val < 0.2:
            reward_text = _color.fail(reward_text)
        else:
            reward_text = _color.warn(reward_text)
        print(f"Reward: {reward_text}")
        exec_status = result['execution']['status']
        if exec_status == 'executed':
            exec_text = _color.ok(exec_status)
        else:
            exec_text = _color.warn(exec_status)
        print(f"Execution: {exec_text}")
        emergency = result['safety']['emergency_mode']
        safety_text = _color.fail(f"emergency={emergency}") if emergency else _color.ok(f"emergency={emergency}")
        print(f"Safety: {safety_text}")
        if verbose:
            print()
            print(_color.info("--- Verbose Details ---"))
            reward = result.get('reward', {})
            print(f"  Reward breakdown: safe={reward.get('safe_reward', 0):.4f}, "
                  f"gain={reward.get('gain_reward', 0):.4f}, eff={reward.get('eff_reward', 0):.4f}")
            execution = result.get('execution', {})
            print(f"  Execution: task_id={execution.get('task_id', '-')}, "
                  f"success={execution.get('success', '-')}")
            safety = result.get('safety', {})
            print(f"  Safety: cpu={safety.get('cpu_percent', 0):.1f}%, "
                  f"mem={safety.get('mem_percent', 0):.1f}%")
            errors = result.get('errors', {})
            if isinstance(errors, dict):
                print(f"  Error details: cpu={errors.get('cpu_percent', 0):.4f}, "
                      f"mem={errors.get('mem_percent', 0):.4f}, "
                      f"disk={errors.get('disk_percent', 0):.4f}")

    def cmd_quickstart(self, args: argparse.Namespace) -> int:
        """一键快速体验：doctor → config init → cycle → status。

        :param args: 命令行参数。
        :type args: argparse.Namespace
        :return: 全流程成功返回0，任一步失败返回1。
        :rtype: int
        """
        print("=" * SEPARATOR_WIDTH)
        print("  Self-Evolving Agent - 快速体验")
        print("  30秒体验自驱进化 Agent 系统")
        print("=" * SEPARATOR_WIDTH)
        print()

        # Step 1: 健康检查
        print("[1/4] 健康检查 (doctor)")
        print("-" * SEPARATOR_WIDTH)
        doctor_args = argparse.Namespace(config=args.config)
        doctor_ret = self.cmd_doctor(doctor_args)
        if doctor_ret != 0:
            print("\n健康检查失败，请先修复上述问题再继续。")
            return 1
        print()

        # Step 2: 初始化配置（若不存在）
        print("[2/4] 初始化配置 (config init)")
        print("-" * SEPARATOR_WIDTH)
        config_path = args.config or DEFAULT_CONFIG_PATH
        if os.path.exists(config_path):
            print(f"配置已存在: {config_path} (跳过初始化)")
        else:
            init_args = argparse.Namespace(config=config_path, action='init', force=False)
            init_ret = self.cmd_config(init_args)
            if init_ret != 0:
                print("\n配置初始化失败。")
                return 1
        print()

        # Step 3: 执行单次循环
        print("[3/4] 执行单次循环 (cycle)")
        print("-" * SEPARATOR_WIDTH)
        cycle_args = argparse.Namespace(config=config_path, json=False)
        cycle_ret = self.cmd_cycle(cycle_args)
        if cycle_ret != 0:
            print("\n循环执行失败。")
            return 1
        print()

        # Step 4: 查看状态
        print("[4/4] 查看状态 (status)")
        print("-" * SEPARATOR_WIDTH)
        status_args = argparse.Namespace(config=config_path, json=False)
        status_ret = self.cmd_status(status_args)
        if status_ret != 0:
            print("\n状态查询失败。")
            return 1
        print()

        print("=" * SEPARATOR_WIDTH)
        print("  快速体验完成！")
        print("  系统已成功运行一个完整周期。")
        print()
        print("  后续步骤:")
        print("    - 持续运行: self-evolving-agent start --cycles 10")
        print("    - 查看日志: self-evolving-agent logs --lines 20")
        print("    - 查看文档: README.md")
        print("=" * SEPARATOR_WIDTH)
        return 0

    def cmd_doctor(self, args: argparse.Namespace) -> int:
        """健康检查：验证运行环境、依赖、配置、路径权限。

        :param args: 命令行参数。
        :type args: argparse.Namespace
        :return: 全部通过返回0，存在问题返回1。
        :rtype: int
        """
        checks: List[Tuple[str, bool, str]] = []
        checks.extend(self._check_python_version())
        checks.extend(self._check_dependencies())
        checks.extend(self._check_config(args))
        checks.extend(self._check_paths(args))

        passed = sum(1 for _, ok, _ in checks if ok)
        total = len(checks)

        # JSON 输出模式
        if getattr(args, 'json', False):
            result = {
                'version': _get_version(),
                'passed': passed,
                'total': total,
                'all_passed': passed == total,
                'checks': [
                    {'name': name, 'ok': ok, 'message': msg}
                    for name, ok, msg in checks
                ],
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if passed == total else 1

        # 文本输出模式
        print(f"Self-Evolving Agent v{_get_version()} - Health Check")
        print("=" * SEPARATOR_WIDTH)
        for name, ok, msg in checks:
            if ok:
                status = _color.ok("[OK]  ")
            else:
                status = _color.fail("[FAIL]")
            print(f"  {status} {name}: {msg}")

        print("=" * SEPARATOR_WIDTH)
        result_text = f"Result: {passed}/{total} checks passed"
        if passed == total:
            print(_color.ok(result_text))
            print(_color.ok("All checks passed. System is ready."))
            return 0
        print(_color.fail(result_text))
        print(_color.fail("Some checks failed. Please review the issues above."))
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
        except Exception as e:
            logger.warning("Failed to read paths from config, using defaults: %s", e)

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
        except Exception as e:
            logger.warning("Failed to resolve log file path: %s", e)
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

    def cmd_dashboard(self, args: argparse.Namespace) -> int:
        """启动监控仪表盘 Web UI。

        读取 metrics.json 实时展示 Agent 运行状态，无需 Agent 在线。
        用户通过浏览器访问 http://host:port 查看仪表盘。

        :param args: 命令行参数，支持 --host/--port/--auth-token/--auto-port。
        :type args: argparse.Namespace
        :return: 成功启动并正常退出返回0，端口占用等错误返回1。
        :rtype: int
        """
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1

        data_dir = self._resolve_data_dir(config_path)
        host = args.host
        port = args.port
        auth_token = args.auth_token
        auto_port = getattr(args, 'auto_port', False)

        try:
            from src.dashboard.server import _DASHBOARD_HTML
        except ImportError:
            logger.error("Dashboard module not found.")
            return 1

        handler = self._create_standalone_handler(
            _DASHBOARD_HTML, config_path, data_dir, auth_token
        )

        # 端口冲突时自动寻找可用端口（仅当 --auto-port 启用时）
        actual_port = port
        if auto_port:
            actual_port = self._find_available_port(host, port)
            if actual_port != port:
                print(_color.warn(
                    f"Port {port} is in use, using {actual_port} instead."
                ))

        print("Self-Evolving Agent Dashboard")
        print(f"  URL:          http://{host}:{actual_port}")
        print(f"  Config:       {config_path}")
        print(f"  Data dir:     {data_dir}")
        if auth_token:
            print(f"  Auth token:   {auth_token}")
            print(f"  Access API with: curl -H 'X-Auth-Token: {auth_token}' http://{host}:{actual_port}/api/data")
        else:
            print("  Auth:         disabled (bind to 127.0.0.1 for safety)")
        print("  Press Ctrl+C to stop.")
        print()

        try:
            server = HTTPServer((host, actual_port), handler)
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            logger.info("Dashboard stopped by user (KeyboardInterrupt).")
            server.shutdown()
            server.server_close()
        except OSError as e:
            logger.error("Failed to start dashboard on %s:%s: %s", host, actual_port, e)
            print(f"Failed to start dashboard on {host}:{actual_port}: {e}")
            if not auto_port:
                print("Hint: Port may be in use. Try a different port with --port, "
                      "or use --auto-port to find an available one automatically.")
            return 1
        return 0

    @staticmethod
    def _find_available_port(host: str, preferred_port: int) -> int:
        """查找可用端口。优先使用 preferred_port，被占用则由 OS 分配。

        :param host: 绑定的主机。
        :type host: str
        :param preferred_port: 优先使用的端口。
        :type preferred_port: int
        :return: 可用端口号。
        :rtype: int
        """
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, preferred_port))
                return preferred_port
        except OSError as e:
            logger.debug("Preferred port %s unavailable, falling back to OS-assigned: %s",
                         preferred_port, e)
        # 由 OS 分配可用端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]

    def _resolve_data_dir(self, config_path: str) -> str:
        """从配置文件解析数据目录。"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get('paths', {}).get('data_dir', 'data')
        except Exception as e:
            logger.warning("Failed to resolve data_dir, using default 'data': %s", e)
            return 'data'

    def _create_standalone_handler(
        self, html_template: str, config_path: str, data_dir: str,
        auth_token: Optional[str] = None,
    ) -> type:
        """创建独立的 dashboard HTTP 请求处理器类型。

        :param html_template: 仪表盘 HTML 模板字符串。
        :param config_path: 配置文件路径。
        :param data_dir: 数据目录路径。
        :param auth_token: 可选认证令牌，非空时 API 请求需携带 X-Auth-Token 头。
        :return: BaseHTTPRequestHandler 子类。
        """
        cli_ref = self
        config_ref = config_path
        token_ref = auth_token

        class StandaloneDashboardHandler(BaseHTTPRequestHandler):
            """独立仪表盘处理器：每次请求读取 metrics.json 获取实时数据。"""

            def do_GET(self):
                route = self._get_route()
                route()

            def _check_auth(self) -> bool:
                """检查请求是否携带有效认证令牌。"""
                if not token_ref:
                    return True
                provided = self.headers.get('X-Auth-Token', '')
                return provided == token_ref

            def _get_route(self):
                if self.path == '/':
                    return self._serve_html
                elif self.path == '/api/data':
                    return self._serve_data
                elif self.path == '/api/health':
                    return self._serve_health
                return self._send_404

            def _serve_html(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html_template.encode('utf-8'))

            def _serve_data(self):
                if not self._check_auth():
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Unauthorized'}).encode('utf-8'))
                    return
                stats = cli_ref._get_lightweight_status(config_ref)
                history = cli_ref._read_history_from_snapshots()
                data = {
                    'cycles': stats.get('cycle_count', 0),
                    'recent_mae': stats.get('recent_mae', 0.0),
                    'success_rate': stats.get('recent_success_rate', 0.0),
                    'reward': 0.0,
                    'memory_samples': stats.get('memory_samples', 0),
                    'task_pool': {'total_tasks': 0},
                    'predictions': {},
                    'current_state': 'ACTIVE' if stats.get('running') else 'IDLE',
                    'history': history,
                }
                payload = json.dumps(data, default=str).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payload)

            def _serve_health(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok'}).encode('utf-8'))

            def _send_404(self):
                self.send_error(404)

            def log_message(self, format, *args):
                pass

        return StandaloneDashboardHandler

    def _read_history_from_snapshots(self) -> List[Dict[str, Any]]:
        """从快照索引读取历史周期数据，供仪表盘趋势图展示。

        快照索引条目格式：{id, tag, timestamp, filepath, metrics:{mae,coverage,eff_score}}

        :return: 历史记录列表，每项包含 cycle/mae/reward/duration。
        :rtype: List[Dict[str, Any]]
        """
        index_path = os.path.join('snapshots', 'index.json')
        if not os.path.exists(index_path):
            return []
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            if not isinstance(index, list):
                return []
            history: List[Dict[str, Any]] = []
            for i, entry in enumerate(index[-50:]):
                if not isinstance(entry, dict):
                    continue
                metrics = entry.get('metrics', {}) if isinstance(entry.get('metrics'), dict) else {}
                history.append({
                    'cycle': i + 1,
                    'mae': metrics.get('mae', 0.0),
                    'reward': metrics.get('eff_score', 0.0),
                    'duration': 0.0,
                })
            return history
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to read history from snapshots: %s", e)
            return []

    def cmd_cleanup(self, args: argparse.Namespace) -> int:
        """清理旧数据：快照、轮转日志、记忆数据库碎片。

        :param args: 命令行参数，支持 --keep-snapshots/--keep-logs/--vacuum-db/--dry-run。
        :type args: argparse.Namespace
        :return: 成功返回0。
        :rtype: int
        """
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not self._validate_config_path(config_path):
            return 1

        cfg = self._load_config(config_path)
        data_dir = cfg.get('paths', {}).get('data_dir', 'data')
        log_dir = cfg.get('paths', {}).get('log_dir', 'logs')
        snapshot_dir = cfg.get('paths', {}).get('snapshot_dir', 'snapshots')

        keep_snapshots = args.keep_snapshots if args.keep_snapshots is not None \
            else cfg.get('resource_limits', {}).get('max_snapshot_count', 10)
        keep_logs = args.keep_logs if args.keep_logs is not None \
            else cfg.get('logging', {}).get('backup_count', 5)
        dry_run = args.dry_run
        json_output = getattr(args, 'json', False)

        # 收集结构化结果（用于 JSON 输出）
        snap_result = self._cleanup_snapshots(snapshot_dir, keep_snapshots, dry_run)
        logs_result = self._cleanup_logs(log_dir, keep_logs, dry_run)
        db_result = {'enabled': args.vacuum_db, 'freed_bytes': 0,
                     'size_before': 0, 'size_after': 0}
        if args.vacuum_db:
            db_result = self._vacuum_memory_db(data_dir, dry_run)

        total_freed = (snap_result['freed_bytes'] + logs_result['freed_bytes']
                       + db_result.get('freed_bytes', 0))

        # JSON 输出模式
        if json_output:
            result = {
                'dry_run': dry_run,
                'total_freed_bytes': total_freed,
                'total_freed_human': self._format_size(total_freed),
                'categories': {
                    'snapshots': snap_result,
                    'logs': logs_result,
                    'vacuum_db': db_result,
                },
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        # 文本输出模式
        print("=" * SEPARATOR_WIDTH)
        print("Self-Evolving Agent - Data Cleanup")
        if dry_run:
            print("  (DRY RUN - no files will be deleted)")
        print("=" * SEPARATOR_WIDTH)
        print()

        for line in snap_result['details']:
            print(line)
        for line in logs_result['details']:
            print(line)
        if args.vacuum_db:
            for line in db_result.get('details', []):
                print(line)

        print()
        print("-" * SEPARATOR_WIDTH)
        if dry_run:
            print(_color.warn(f"Would free approximately {self._format_size(total_freed)}"))
        else:
            freed_text = f"Freed approximately {self._format_size(total_freed)}"
            if total_freed > 0:
                print(_color.ok(freed_text))
            else:
                print(freed_text)
        print("=" * SEPARATOR_WIDTH)
        return 0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件，失败时返回空字典并记录警告。"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load config %s: %s", config_path, e)
            return {}

    def _cleanup_snapshots(self, snapshot_dir: str, keep: int,
                           dry_run: bool) -> Dict[str, Any]:
        """清理旧快照文件，保留最近 N 个，并清除孤立文件。

        :param snapshot_dir: 快照目录。
        :param keep: 保留的快照数量。
        :param dry_run: 仅模拟不实际删除。
        :return: 结构化清理结果（freed_bytes/details/orphaned_removed/indexed_removed/kept）。
        :rtype: Dict[str, Any]
        """
        index_path = os.path.join(snapshot_dir, 'index.json')
        details: List[str] = []
        if not os.path.exists(snapshot_dir):
            details.append(f"[snapshots] Directory not found: {snapshot_dir}")
            return {'freed_bytes': 0, 'details': details,
                    'orphaned_removed': 0, 'indexed_removed': 0, 'kept': 0}

        # 收集磁盘上所有 .pkl 文件
        all_pkl = set(
            f for f in os.listdir(snapshot_dir) if f.endswith('.pkl')
        )

        entries = self._load_snapshot_index(index_path)
        if entries:
            entries_sorted = sorted(entries, key=lambda e: e.get('timestamp', 0))
            indexed_files = [os.path.basename(e.get('filepath', '')) for e in entries_sorted]
            indexed_files = [f for f in indexed_files if f]
        else:
            indexed_files = sorted(
                all_pkl,
                key=lambda f: os.path.getmtime(os.path.join(snapshot_dir, f))
            )

        # 识别孤立文件（在磁盘上但不在索引中）
        indexed_set = set(indexed_files)
        orphaned = all_pkl - indexed_set

        freed = 0
        orphaned_removed = 0
        if orphaned:
            details.append(f"[snapshots] {len(orphaned)} orphaned files (not in index).")
            for fname in sorted(orphaned):
                fpath = os.path.join(snapshot_dir, fname)
                size = os.path.getsize(fpath)
                if not dry_run:
                    os.remove(fpath)
                freed += size
                orphaned_removed += 1
                details.append(f"  {'[DRY] ' if dry_run else ''}removed orphan: {fname} "
                               f"({self._format_size(size)})")

        if len(indexed_files) <= keep:
            details.append(f"[snapshots] {len(indexed_files)} indexed files, keep={keep}, "
                           f"nothing to trim.")
            return {'freed_bytes': freed, 'details': details,
                    'orphaned_removed': orphaned_removed, 'indexed_removed': 0,
                    'kept': len(indexed_files)}

        to_remove = indexed_files[:-keep] if keep > 0 else indexed_files
        details.append(f"[snapshots] {len(indexed_files)} indexed files, keeping {keep}, "
                       f"removing {len(to_remove)}.")

        indexed_removed = 0
        for fname in to_remove:
            fpath = os.path.join(snapshot_dir, fname)
            if not os.path.exists(fpath):
                continue
            size = os.path.getsize(fpath)
            if not dry_run:
                os.remove(fpath)
            freed += size
            indexed_removed += 1
            details.append(f"  {'[DRY] ' if dry_run else ''}removed: {fname} "
                           f"({self._format_size(size)})")

        if not dry_run and to_remove:
            kept_entries = entries[-keep:] if isinstance(entries, list) and entries else []
            if kept_entries:
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(kept_entries, f, indent=2)
                details.append(f"  updated index.json: {len(kept_entries)} entries remain.")

        return {'freed_bytes': freed, 'details': details,
                'orphaned_removed': orphaned_removed,
                'indexed_removed': indexed_removed,
                'kept': max(len(indexed_files) - indexed_removed, 0)}

    def _load_snapshot_index(self, index_path: str) -> List[Dict[str, Any]]:
        """加载快照索引文件。"""
        if not os.path.exists(index_path):
            return []
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to load snapshot index from %s: %s", index_path, e)
            return []

    def _cleanup_logs(self, log_dir: str, keep: int,
                      dry_run: bool) -> Dict[str, Any]:
        """清理轮转的旧日志文件，保留最近 N 个。

        :param log_dir: 日志目录。
        :param keep: 保留的轮转日志数量（不含当前 agent.log）。
        :param dry_run: 仅模拟不实际删除。
        :return: 结构化清理结果（freed_bytes/details/rotated_total/removed/kept）。
        :rtype: Dict[str, Any]
        """
        details: List[str] = []
        if not os.path.exists(log_dir):
            details.append(f"[logs] Directory not found: {log_dir}")
            return {'freed_bytes': 0, 'details': details,
                    'rotated_total': 0, 'removed': 0, 'kept': 0}

        rotated = []
        for fname in os.listdir(log_dir):
            if fname.startswith('agent.log.') and not fname.endswith('.lck'):
                fpath = os.path.join(log_dir, fname)
                if os.path.isfile(fpath):
                    rotated.append((fname, fpath, os.path.getmtime(fpath)))

        if not rotated:
            details.append(f"[logs] No rotated log files found in {log_dir}.")
            return {'freed_bytes': 0, 'details': details,
                    'rotated_total': 0, 'removed': 0, 'kept': 0}

        rotated.sort(key=lambda x: x[2], reverse=True)
        to_remove = rotated[keep:] if keep > 0 else rotated
        details.append(f"[logs] {len(rotated)} rotated files, keeping {keep}, "
                       f"removing {len(to_remove)}.")

        freed = 0
        removed_count = 0
        for fname, fpath, _ in to_remove:
            size = os.path.getsize(fpath)
            if not dry_run:
                os.remove(fpath)
            freed += size
            removed_count += 1
            details.append(f"  {'[DRY] ' if dry_run else ''}removed: {fname} "
                           f"({self._format_size(size)})")

        return {'freed_bytes': freed, 'details': details,
                'rotated_total': len(rotated),
                'removed': removed_count,
                'kept': len(rotated) - removed_count}

    def _vacuum_memory_db(self, data_dir: str,
                          dry_run: bool) -> Dict[str, Any]:
        """对记忆数据库执行 VACUUM 回收碎片空间。

        :param data_dir: 数据目录。
        :param dry_run: 仅模拟不实际执行。
        :return: 结构化清理结果（enabled/freed_bytes/size_before/size_after/details）。
        :rtype: Dict[str, Any]
        """
        details: List[str] = []
        db_path = os.path.join(data_dir, 'memory.db')
        if not os.path.exists(db_path):
            details.append(f"[db] Memory database not found: {db_path}")
            return {'enabled': True, 'freed_bytes': 0, 'size_before': 0,
                    'size_after': 0, 'details': details}

        size_before = os.path.getsize(db_path)
        details.append(f"[db] Memory database: {self._format_size(size_before)}")

        if dry_run:
            details.append("  [DRY] would VACUUM (estimate ~10-30% reclaimable)")
            return {'enabled': True, 'freed_bytes': 0, 'size_before': size_before,
                    'size_after': size_before, 'details': details}

        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute('VACUUM')
            conn.close()
            size_after = os.path.getsize(db_path)
            freed = size_before - size_after
            details.append(f"  VACUUM complete: {self._format_size(size_before)} -> "
                           f"{self._format_size(size_after)} "
                           f"(freed {self._format_size(freed)})")
            return {'enabled': True, 'freed_bytes': max(freed, 0),
                    'size_before': size_before, 'size_after': size_after,
                    'details': details}
        except sqlite3.Error as e:
            details.append(f"  VACUUM failed: {e}")
            return {'enabled': True, 'freed_bytes': 0, 'size_before': size_before,
                    'size_after': size_before, 'details': details, 'error': str(e)}

    def _format_size(self, size: int) -> str:
        """将字节数格式化为人类可读的大小字符串。"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


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
示例:
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
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output (detailed cycle info)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')

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
    config_parser.add_argument('action', choices=['show', 'validate', 'init', 'wizard'])
    config_parser.add_argument('--force', action='store_true', help='Force overwrite')

    # stop
    subparsers.add_parser('stop', help='Stop the agent')

    # cycle
    subparsers.add_parser('cycle', help='Run a single cycle')

    # doctor
    subparsers.add_parser('doctor', help='Run health check (deps, config, paths)')

    # logs
    logs_parser = subparsers.add_parser('logs', help='Show recent log lines')
    logs_parser.add_argument('--lines', '-l', type=int, default=50,
                             help='Number of lines to show (default: 50)')

    # quickstart
    subparsers.add_parser('quickstart', help='One-click demo: doctor + init + cycle + status')

    # dashboard
    dashboard_parser = subparsers.add_parser(
        'dashboard', help='Start monitoring dashboard web UI')
    dashboard_parser.add_argument('--host', default='127.0.0.1',
                                  help='Dashboard host (default: 127.0.0.1)')
    dashboard_parser.add_argument('--port', '-p', type=int, default=8080,
                                  help='Dashboard port (default: 8080)')
    dashboard_parser.add_argument('--auth-token', default=None,
                                  help='Auth token for API access (X-Auth-Token header)')
    dashboard_parser.add_argument('--auto-port', action='store_true',
                                  help='Automatically find an available port if the specified one is in use')

    # cleanup
    cleanup_parser = subparsers.add_parser(
        'cleanup', help='Clean up old snapshots, rotated logs, vacuum DB')
    cleanup_parser.add_argument('--keep-snapshots', type=int, default=None,
                                help='Number of snapshots to keep (default: from config)')
    cleanup_parser.add_argument('--keep-logs', type=int, default=None,
                                help='Number of rotated logs to keep (default: from config)')
    cleanup_parser.add_argument('--vacuum-db', action='store_true',
                                help='Vacuum memory database to reclaim space')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                                help='Show what would be cleaned without deleting')

    args = parser.parse_args()

    if getattr(args, 'no_color', False):
        _color.disable()

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
        'quickstart': cli.cmd_quickstart,
        'dashboard': cli.cmd_dashboard,
        'cleanup': cli.cmd_cleanup,
    }

    handler = command_map.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
