#!/usr/bin/env python3
"""系统服务守护进程管理器"""

import abc
import os
import sys
import time
import signal
import atexit
import logging
from typing import Optional, Callable, List, Any, Dict

from .constants import DAEMON_PID_FILE, DAEMON_WORKING_DIR, DAEMON_UMASK

logger = logging.getLogger(__name__)


class BaseDaemonManager(abc.ABC):
    """守护进程管理器抽象基类"""

    def __init__(self, pid_file: str = DAEMON_PID_FILE,
                 working_dir: str = DAEMON_WORKING_DIR,
                 umask: int = DAEMON_UMASK) -> None:
        """初始化守护进程管理器。

        :param pid_file: PID文件路径。
        :type pid_file: str
        :param working_dir: 工作目录。
        :type working_dir: str
        :param umask: 文件权限掩码。
        :type umask: int
        :return: 无。
        :rtype: None
        """
        self.pid_file = pid_file
        self.working_dir = working_dir
        self.umask = umask
        self._reset_state()

    def _reset_state(self) -> None:
        self._running = False
        self._worker: Optional[Callable[[], Any]] = None
        self._shutdown_hooks: List[Callable[[], Any]] = []

    def _init_daemon_state(self) -> None:
        self._running = False
        self._worker: Optional[Callable[[], Any]] = None
        self._shutdown_hooks: List[Callable[[], Any]] = []

    @abc.abstractmethod
    def register_worker(self, worker: Callable) -> None:
        """注册主工作函数。

        :param worker: 工作函数。
        :type worker: Callable
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def register_shutdown_hook(self, hook: Callable) -> None:
        """注册关闭钩子。

        :param hook: 关闭钩子函数。
        :type hook: Callable
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def start(self) -> None:
        """启动守护进程。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def stop(self) -> bool:
        """停止守护进程。

        :param: 无
        :type: 无
        :return: 是否成功停止。
        :rtype: bool
        """
        ...

    @abc.abstractmethod
    def restart(self) -> None:
        """重启守护进程。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def status(self) -> Dict[str, Any]:
        """获取守护进程状态。

        :param: 无
        :type: 无
        :return: 状态字典。
        :rtype: Dict[str, Any]
        """
        ...

    @abc.abstractmethod
    def is_running(self) -> bool:
        """检查守护进程是否运行中。

        :param: 无
        :type: 无
        :return: 是否运行中。
        :rtype: bool
        """
        ...

    def _daemonize(self) -> None:
        """执行Unix守护进程化（双fork）"""
        self._fork_parent()
        self._setup_session()
        self._fork_parent_again()
        self._redirect_streams()
        self._setup_pid_file()
        self._setup_signals()

    def _fork_parent(self) -> None:
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            logger.error("Fork #1 failed: %s", e)
            sys.exit(1)

    def _setup_session(self) -> None:
        os.chdir(self.working_dir)
        os.setsid()
        os.umask(self.umask)

    def _fork_parent_again(self) -> None:
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            logger.error("Fork #2 failed: %s", e)
            sys.exit(1)

    def _redirect_streams(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        with open('/dev/null', 'r') as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open('/dev/null', 'a+') as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
            os.dup2(f.fileno(), sys.stderr.fileno())

    def _setup_pid_file(self) -> None:
        self._write_pid(os.getpid())
        atexit.register(self._remove_pid)

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """信号处理器"""
        logger.info("Received signal %s, shutting down...", signum)
        self._running = False

    def _run(self) -> None:
        """主运行循环"""
        self._running = True
        logger.info("Daemon started (PID: %s)", os.getpid())
        self._execute_worker()
        self._shutdown()

    def _execute_worker(self) -> None:
        if self._worker:
            try:
                self._worker()
            except Exception as e:
                logger.error("Worker error: %s", e)

    def _shutdown(self) -> None:
        """执行关闭流程"""
        logger.info("Daemon shutting down...")
        self._run_shutdown_hooks()
        self._remove_pid()
        logger.info("Daemon stopped")

    def _run_shutdown_hooks(self) -> None:
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception as e:
                logger.error("Shutdown hook error: %s", e)

    def _write_pid(self, pid: int) -> None:
        """写入PID文件"""
        os.makedirs(os.path.dirname(self.pid_file) or '.', exist_ok=True)
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))

    def _read_pid(self) -> Optional[int]:
        """读取PID文件"""
        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            return None

    def _remove_pid(self) -> None:
        """删除PID文件"""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
        except OSError as e:
            logger.warning("Daemon._remove_pid: remove %s failed: %s", self.pid_file, e)

    def _is_process_alive(self, pid: int) -> bool:
        """检查进程是否存活"""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


class DaemonManager(BaseDaemonManager):
    """轻量级守护进程管理器"""

    def __init__(self, pid_file: str = DAEMON_PID_FILE,
                 working_dir: str = DAEMON_WORKING_DIR,
                 umask: int = DAEMON_UMASK) -> None:
        """初始化守护进程管理器。

        :param pid_file: PID文件路径。
        :type pid_file: str
        :param working_dir: 工作目录。
        :type working_dir: str
        :param umask: 文件权限掩码。
        :type umask: int
        :return: 无。
        :rtype: None
        """
        super().__init__(pid_file, working_dir, umask)

    def register_worker(self, worker: Callable[[], Any]) -> None:
        """注册主工作函数。

        :param worker: 工作函数。
        :type worker: Callable[[], Any]
        :return: 无。
        :rtype: None
        """
        self._worker = worker

    def register_shutdown_hook(self, hook: Callable[[], Any]) -> None:
        """注册关闭钩子。

        :param hook: 关闭钩子函数。
        :type hook: Callable[[], Any]
        :return: 无。
        :rtype: None
        """
        self._shutdown_hooks.append(hook)

    def start(self) -> None:
        """启动守护进程。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        if self.is_running():
            logger.error("Daemon already running (PID: %s)", self._read_pid())
            sys.exit(1)

        self._daemonize()
        self._run()

    def stop(self) -> bool:
        """停止守护进程。

        :param: 无
        :type: 无
        :return: 是否成功停止。
        :rtype: bool
        """
        pid = self._read_pid()
        if not pid:
            logger.warning("Daemon not running")
            return False
        return self._do_stop(pid)

    def _do_stop(self, pid: int) -> bool:
        try:
            self._send_stop_signal(pid)
            self._wait_for_stop(pid)
            self._remove_pid()
            logger.info("Daemon stopped (PID: %s)", pid)
            return True
        except ProcessLookupError:
            self._remove_pid()
            return True
        except Exception as e:
            return self._handle_stop_error(e)

    def _handle_stop_error(self, e: Exception) -> bool:
        logger.error("Stop daemon error: %s", e)
        return False

    def _send_stop_signal(self, pid: int) -> None:
        os.kill(pid, signal.SIGTERM)

    def _wait_for_stop(self, pid: int) -> None:
        for _ in range(30):
            if not self.is_running():
                break
            time.sleep(0.5)
        if self.is_running():
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

    def restart(self) -> None:
        """重启守护进程。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self.stop()
        time.sleep(1)
        self.start()

    def status(self) -> dict:
        """获取守护进程状态。

        :param: 无
        :type: 无
        :return: 状态字典。
        :rtype: dict
        """
        pid = self._read_pid()
        if pid and self._is_process_alive(pid):
            return self._build_status_dict(True, pid)
        return self._build_status_dict(False, None)

    def _build_status_dict(self, running: bool, pid: Optional[int]) -> Dict[str, Any]:
        return {
            'running': running,
            'pid': pid,
            'pid_file': self.pid_file,
        }

    def is_running(self) -> bool:
        """检查守护进程是否运行中。

        :param: 无
        :type: 无
        :return: 是否运行中。
        :rtype: bool
        """
        pid = self._read_pid()
        return pid is not None and self._is_process_alive(pid)


def run_daemon_command(command: str, worker: Callable,
                       pid_file: str = DAEMON_PID_FILE) -> int:
    """便捷函数：执行守护进程命令。

    :param command: 命令名称（start/stop/restart/status）。
    :type command: str
    :param worker: 工作函数。
    :type worker: Callable
    :param pid_file: PID文件路径。
    :type pid_file: str
    :return: 退出码。
    :rtype: int
    """
    daemon = DaemonManager(pid_file=pid_file)
    daemon.register_worker(worker)
    return _handle_command(daemon, command)


def _handle_command(daemon: DaemonManager, command: str) -> int:
    if command == 'start':
        return _handle_start(daemon)
    elif command == 'stop':
        return _handle_stop(daemon)
    elif command == 'restart':
        daemon.restart()
        return 0
    elif command == 'status':
        return _handle_status(daemon)
    else:
        logger.error("Unknown command: %s", command)
        return 1


def _handle_start(daemon: DaemonManager) -> int:
    if daemon.is_running():
        logger.info("Daemon already running (PID: %s)", daemon._read_pid())
        return 1
    daemon.start()
    return 0


def _handle_stop(daemon: DaemonManager) -> int:
    if daemon.stop():
        logger.info("Daemon stopped")
        return 0
    logger.info("Daemon not running")
    return 1


def _handle_status(daemon: DaemonManager) -> int:
    status = daemon.status()
    if status['running']:
        logger.info("Daemon running (PID: %s)", status['pid'])
    else:
        logger.info("Daemon not running")
    return 0
