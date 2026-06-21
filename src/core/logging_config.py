"""日志配置组件。

从 agent_loop.py 提取的日志系统配置逻辑，遵循单一职责原则：
将日志处理器创建、根日志器配置、日志处理器清理等关注点从主循环中分离。

支持特性：
- 文本/JSON 两种日志格式（通过 logging.format_type 配置）
- 按模块级别配置日志（通过 logging.loggers 映射配置）

:author: self_evolving_agent
:since: 阶段158
"""

import json
import logging
import logging.handlers
import os
import sys
from typing import Any, Dict, Optional


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """安全的日志文件轮转处理器。

    在磁盘满、IO错误等场景下捕获OSError，避免日志写入异常导致应用崩溃，
    同时向标准错误输出简要提示。
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except OSError as e:
            try:
                sys.stderr.write(f"[LOGGING ERROR] failed to write log: {e}\n")
            except OSError:
                pass

    def handleError(self, record: logging.LogRecord) -> None:
        """覆盖默认错误处理，防止异常上抛。"""
        try:
            sys.stderr.write(f"[LOGGING ERROR] handler error for record {record.name}\n")
        except OSError:
            pass


class JsonFormatter(logging.Formatter):
    """JSON 格式日志格式化器。

    将日志记录序列化为单行 JSON，便于 ELK/Loki/CloudWatch 等
    日志聚合系统进行结构化检索与字段过滤。
    """

    def format(self, record: logging.LogRecord) -> str:
        """将日志记录格式化为 JSON 字符串。

        :param record: 日志记录对象。
        :type record: logging.LogRecord
        :return: JSON 格式的日志字符串。
        :rtype: str
        """
        log_entry: Dict[str, Any] = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'func': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process,
        }
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class LoggingConfigurator:
    """日志系统配置器。

    负责创建安全的轮转文件处理器、配置根日志器级别与格式、
    以及在关闭时清理已注册的处理器。所有方法均为静态方法，
    不持有状态，便于在任意上下文中调用。
    """

    @staticmethod
    def setup(cfg: Dict[str, Any]) -> SafeRotatingFileHandler:
        """根据配置创建日志处理器并配置根日志器。

        :param cfg: 完整配置字典。
        :type cfg: Dict[str, Any]
        :return: 已注册到根日志器的日志处理器。
        :rtype: SafeRotatingFileHandler
        """
        log_cfg = cfg.get('logging', {})
        level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
        log_dir = cfg.get('paths', {}).get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        handler = LoggingConfigurator.create_log_handler(log_cfg, log_dir)
        LoggingConfigurator.configure_root_logger(level, handler)
        LoggingConfigurator.configure_module_loggers(log_cfg)
        return handler

    @staticmethod
    def create_log_handler(log_cfg: Dict[str, Any], log_dir: str) -> SafeRotatingFileHandler:
        """创建轮转文件日志处理器。

        :param log_cfg: 日志配置字典。
        :type log_cfg: Dict[str, Any]
        :param log_dir: 日志目录。
        :type log_dir: str
        :return: 日志处理器实例。
        :rtype: SafeRotatingFileHandler
        """
        log_file = os.path.join(log_dir, log_cfg.get('file', 'agent.log').split('/')[-1])
        handler = SafeRotatingFileHandler(
            log_file,
            maxBytes=log_cfg.get('max_bytes', 10485760),
            backupCount=log_cfg.get('backup_count', 5)
        )
        handler.setFormatter(LoggingConfigurator.create_formatter(log_cfg))
        return handler

    @staticmethod
    def create_formatter(log_cfg: Dict[str, Any]) -> logging.Formatter:
        """创建日志格式化器。

        根据 format_type 配置决定使用文本格式还是 JSON 格式：
        - text（默认）：使用 format 字段指定的文本格式
        - json：使用 JsonFormatter 输出结构化 JSON 日志

        :param log_cfg: 日志配置字典。
        :type log_cfg: Dict[str, Any]
        :return: 格式化器实例。
        :rtype: logging.Formatter
        """
        format_type = log_cfg.get('format_type', 'text').lower()
        if format_type == 'json':
            return JsonFormatter()
        fmt = log_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.Formatter(fmt)

    @staticmethod
    def configure_root_logger(level: int, handler: SafeRotatingFileHandler) -> None:
        """配置根日志器级别并注册处理器。

        :param level: 日志级别。
        :type level: int
        :param handler: 日志处理器。
        :type handler: SafeRotatingFileHandler
        :return: 无。
        :rtype: None
        """
        root = logging.getLogger()
        root.setLevel(level)
        LoggingConfigurator.close_existing_file_handlers(root)
        root.addHandler(handler)

    @staticmethod
    def configure_module_loggers(log_cfg: Dict[str, Any]) -> None:
        """按模块级别配置日志器。

        根据 logging.loggers 映射为指定模块设置独立日志级别，
        实现如 src.core=DEBUG, src.cli=INFO 的差异化配置。
        模块日志器默认 propagate=True，日志仍会传播到 root handler。

        :param log_cfg: 日志配置字典，读取其中的 loggers 字段。
        :type log_cfg: Dict[str, Any]
        :return: 无。
        :rtype: None
        """
        loggers_cfg = log_cfg.get('loggers', {})
        if not isinstance(loggers_cfg, dict):
            return
        for name, lvl_name in loggers_cfg.items():
            lvl = getattr(logging, str(lvl_name).upper(), None)
            if lvl is not None:
                logging.getLogger(name).setLevel(lvl)

    @staticmethod
    def close_existing_file_handlers(root: logging.Logger) -> None:
        """关闭并移除根日志器上已有的文件处理器。

        :param root: 根日志器。
        :type root: logging.Logger
        :return: 无。
        :rtype: None
        """
        for h in list(root.handlers):
            if isinstance(h, (logging.handlers.RotatingFileHandler, SafeRotatingFileHandler)):
                root.removeHandler(h)
                h.close()

    @staticmethod
    def remove_log_handler(handler: Optional[SafeRotatingFileHandler]) -> None:
        """从根日志器移除并关闭指定处理器。

        :param handler: 要移除的日志处理器，为None时无操作。
        :type handler: Optional[SafeRotatingFileHandler]
        :return: 无。
        :rtype: None
        """
        if handler is None:
            return
        root = logging.getLogger()
        if handler in root.handlers:
            root.removeHandler(handler)
        handler.close()
