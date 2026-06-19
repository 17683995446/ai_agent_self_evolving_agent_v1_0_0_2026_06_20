import abc
import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _LogPattern(abc.ABC):
    """日志异常模式定义"""
    name: str
    pattern: str
    severity: str
    threshold: int = 1
    window_seconds: float = 60.0

    @abc.abstractmethod
    def match(self, log_line: str) -> bool:
        """检查日志行是否匹配该模式"""

    def _do_match(self, log_line: str) -> bool:
        return bool(re.search(self.pattern, log_line, re.IGNORECASE))


class LogPattern(_LogPattern):
    """LogPattern 实现"""

    def match(self, log_line: str) -> bool:
        """检查日志行是否匹配该模式"""
        return self._do_match(log_line)


@dataclass
class _AnomalyEvent(abc.ABC):
    """异常事件"""
    pattern_name: str
    severity: str
    message: str
    count: int
    timestamp: float = field(default_factory=time.time)

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""

    def _do_to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_name': self.pattern_name,
            'severity': self.severity,
            'message': self.message,
            'count': self.count,
            'timestamp': self.timestamp,
        }


class AnomalyEvent(_AnomalyEvent):
    """AnomalyEvent 实现"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._do_to_dict()


class _LogAggregator(abc.ABC):
    """日志聚合与异常检测器"""

    DEFAULT_PATTERNS = [
        LogPattern('error_spike', r'ERROR', 'warning', threshold=5, window_seconds=60.0),
        LogPattern('task_failure', r'Task.*failed', 'warning', threshold=3, window_seconds=120.0),
        LogPattern('emergency_mode', r'Emergency mode', 'critical', threshold=1, window_seconds=300.0),
        LogPattern('rollback_triggered', r'rollback', 'critical', threshold=1, window_seconds=300.0),
        LogPattern('resource_violation', r'violation', 'warning', threshold=2, window_seconds=60.0),
    ]

    def __init__(self, log_file: Optional[str] = None,
                 patterns: Optional[List[_LogPattern]] = None,
                 history_size: int = 1000) -> None:
        self.log_file = log_file
        self.patterns = patterns or list(self.DEFAULT_PATTERNS)
        self.history_size = history_size

        self._logs: deque = deque(maxlen=history_size)
        self._matches: Dict[str, deque] = {
            p.name: deque(maxlen=history_size) for p in self.patterns
        }
        self._anomaly_handlers: List[Callable[[_AnomalyEvent], None]] = []
        self._last_check_time: float = 0.0

    @abc.abstractmethod
    def ingest(self, log_line: str) -> None:
        """摄入单条日志"""

    def _do_ingest(self, log_line: str) -> None:
        entry = {
            'message': log_line,
            'timestamp': time.time(),
        }
        self._logs.append(entry)
        for pattern in self.patterns:
            if pattern.match(log_line):
                self._matches[pattern.name].append(entry)

    def ingest_batch(self, log_lines: List[str]) -> None:
        """批量摄入日志"""
        for line in log_lines:
            self.ingest(line)

    def read_from_file(self, lines: int = 100) -> List[str]:
        """从日志文件读取最近N行"""
        if not self.log_file or not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:]]
        except Exception as e:
            logger.error(f"Read log file error: {e}")
            return []

    @abc.abstractmethod
    def check_anomalies(self) -> List[_AnomalyEvent]:
        """检查异常模式"""

    def _do_check_anomalies(self) -> List[_AnomalyEvent]:
        now = time.time()
        anomalies = []
        for pattern in self.patterns:
            event = self._check_pattern(pattern, now)
            if event:
                anomalies.append(event)
                self._notify_anomaly_handlers(event)
        self._last_check_time = now
        return anomalies

    def _check_pattern(self, pattern: _LogPattern, now: float) -> Optional[_AnomalyEvent]:
        matches = self._matches[pattern.name]
        if not matches:
            return None
        recent = [m for m in matches
                  if now - m['timestamp'] <= pattern.window_seconds]
        if len(recent) >= pattern.threshold:
            return AnomalyEvent(
                pattern_name=pattern.name,
                severity=pattern.severity,
                message=f"Pattern '{pattern.name}' triggered {len(recent)} times in {pattern.window_seconds}s",
                count=len(recent),
            )
        return None

    def _notify_anomaly_handlers(self, event: _AnomalyEvent) -> None:
        for handler in self._anomaly_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Anomaly handler error: {e}")

    def on_anomaly(self, handler: Callable[[_AnomalyEvent], None]) -> None:
        """注册异常处理器"""
        self._anomaly_handlers.append(handler)

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合统计"""
        return {
            'total_logs_ingested': len(self._logs),
            'patterns': self._build_pattern_stats(),
            'last_check': self._last_check_time,
        }

    def _build_pattern_stats(self) -> Dict[str, Dict[str, Any]]:
        pattern_stats = {}
        now = time.time()
        for pattern in self.patterns:
            matches = self._matches[pattern.name]
            recent = [m for m in matches
                      if now - m['timestamp'] <= pattern.window_seconds]
            pattern_stats[pattern.name] = {
                'total_matches': len(matches),
                'recent_matches': len(recent),
                'threshold': pattern.threshold,
                'triggered': len(recent) >= pattern.threshold,
            }
        return pattern_stats

    def get_recent_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        """获取最近日志"""
        return list(self._logs)[-count:]

    def clear(self) -> None:
        """清空所有数据"""
        self._logs.clear()
        for dq in self._matches.values():
            dq.clear()


class LogAggregator(_LogAggregator):
    """LogAggregator 实现"""

    def ingest(self, log_line: str) -> None:
        """摄入单条日志"""
        self._do_ingest(log_line)

    def check_anomalies(self) -> List[_AnomalyEvent]:
        """检查异常模式"""
        return self._do_check_anomalies()
