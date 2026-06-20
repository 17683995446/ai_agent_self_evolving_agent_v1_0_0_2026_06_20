import abc
import os
import time
import threading
import logging
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseConfigReloader(abc.ABC):
    """配置文件热重载器基类"""

    def __init__(self, config_path: str, check_interval: float = 5.0):
        """初始化热重载器。

        :param config_path: 配置文件路径。
        :type config_path: str
        :param check_interval: 检查间隔（秒）。
        :type check_interval: float
        :return: 无。
        :rtype: None
        """
        self.config_path = config_path
        self.check_interval = check_interval
        self._last_mtime: float = 0.0
        self._last_content_hash: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list = []
        self._lock = threading.Lock()

        # 初始化时记录文件状态
        self._update_file_state()

    @abc.abstractmethod
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加配置变化回调。

        :param callback: 回调函数。
        :type callback: Callable[[Dict[str, Any]], None]
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """移除配置变化回调。

        :param callback: 要移除的回调函数。
        :type callback: Callable[[Dict[str, Any]], None]
        :return: 是否成功移除。
        :rtype: bool
        """

    @abc.abstractmethod
    def start(self) -> None:
        """启动热重载监听。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """停止热重载监听。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    def force_reload(self) -> Optional[Dict[str, Any]]:
        """强制立即重载配置。

        :param: 无
        :type: 无
        :return: 新配置字典或 None。
        :rtype: Optional[Dict[str, Any]]
        """

    @abc.abstractmethod
    def is_running(self) -> bool:
        """返回是否正在运行。

        :param: 无
        :type: 无
        :return: 是否运行中。
        :rtype: bool
        """

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """

    def _update_file_state(self) -> bool:
        """更新文件状态，返回是否发生变化"""
        try:
            current_mtime = os.path.getmtime(self.config_path)
            with open(self.config_path, 'rb') as f:
                content = f.read()
            current_hash = hash(content)

            with self._lock:
                changed = (
                    current_mtime != self._last_mtime or
                    current_hash != self._last_content_hash
                )
                self._last_mtime = current_mtime
                self._last_content_hash = current_hash
                return changed
        except (OSError, IOError):
            return False


class ConfigReloader(BaseConfigReloader):
    """配置文件热重载器：监听文件变化并自动重载"""

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加配置变化回调。

        :param callback: 回调函数。
        :type callback: Callable[[Dict[str, Any]], None]
        :return: 无。
        :rtype: None
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """移除配置变化回调。

        :param callback: 要移除的回调函数。
        :type callback: Callable[[Dict[str, Any]], None]
        :return: 是否成功移除。
        :rtype: bool
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False

    def start(self) -> None:
        """启动热重载监听。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("ConfigReloader started watching: %s", self.config_path)

    def stop(self) -> None:
        """停止热重载监听。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.check_interval + 1)
        logger.info("ConfigReloader stopped")

    def _watch_loop(self) -> None:
        """文件监听主循环"""
        while self._running:
            try:
                if self._update_file_state():
                    self._on_config_changed()
            except Exception as e:
                logger.error("Config reload error: %s", e)

            time.sleep(self.check_interval)

    def _on_config_changed(self) -> None:
        """配置发生变化时的处理"""
        logger.info("Config file changed: %s", self.config_path)
        new_config = self._load_config_file()
        if new_config is None:
            return
        self._notify_callbacks(new_config)

    def _load_config_file(self) -> Optional[Dict[str, Any]]:
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to parse config: %s", e)
            return None

    def _notify_callbacks(self, new_config: Dict[str, Any]) -> None:
        for callback in self._callbacks:
            try:
                callback(new_config)
            except Exception as e:
                logger.error("Config callback error: %s", e)

    def force_reload(self) -> Optional[Dict[str, Any]]:
        """强制立即重载配置。

        :param: 无
        :type: 无
        :return: 新配置字典或 None。
        :rtype: Optional[Dict[str, Any]]
        """
        new_config = self._load_config_file()
        if new_config is None:
            return None
        self._update_file_state()
        self._notify_callbacks(new_config)
        return new_config

    def is_running(self) -> bool:
        """返回是否正在运行。

        :param: 无
        :type: 无
        :return: 是否运行中。
        :rtype: bool
        """
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """
        return {
            'running': self._running,
            'config_path': self.config_path,
            'check_interval': self.check_interval,
            'callback_count': len(self._callbacks),
            'last_mtime': self._last_mtime,
        }
