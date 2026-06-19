import abc
import json
import logging
import os
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _Alert(abc.ABC):
    """告警记录"""
    alert_id: str
    severity: str
    source: str
    message: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""

    def _do_to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'severity': self.severity,
            'source': self.source,
            'message': self.message,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged,
        }


class Alert(_Alert):
    """Alert 实现"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._do_to_dict()


class _AlertManager(abc.ABC):
    """告警管理器 - 多通道通知与冷却控制"""

    COOLDOWN_SECONDS: Dict[str, float] = {
        'critical': 60.0,
        'warning': 300.0,
        'info': 600.0,
    }

    def __init__(self, alert_log_path: Optional[str] = None) -> None:
        """Initialize alert manager.

        :param alert_log_path: 告警日志文件路径
        :type alert_log_path: Optional[str]
        :return: 无
        :rtype: None
        """
        self.alert_log_path = alert_log_path
        self._alerts: List[_Alert] = []
        self._lock = threading.RLock()
        self._last_alert_time: Dict[str, float] = {}
        self._handlers: List[Callable[[_Alert], None]] = []
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """注册默认处理器。

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        self._handlers.append(self._log_handler)
        if self.alert_log_path:
            self._handlers.append(self._file_handler)

    def add_handler(self, handler: Callable[[_Alert], None]) -> None:
        """添加自定义告警处理器。

        :param handler: 处理器函数
        :type handler: Callable[[_Alert], None]
        :return: 无
        :rtype: None
        """
        with self._lock:
            self._handlers.append(handler)

    def trigger(self, severity: str, source: str, message: str) -> Optional[_Alert]:
        """触发告警，受冷却期控制。

        :param severity: 严重级别 (critical/warning/info)
        :type severity: str
        :param source: 告警来源模块
        :type source: str
        :param message: 告警消息
        :type message: str
        :return: 告警对象或 None（冷却中）
        :rtype: Optional[_Alert]
        """
        with self._lock:
            if not self._is_cooldown_passed(severity, source):
                return None
            alert = self._create_alert(severity, source, message)
            self._alerts.append(alert)
            self._last_alert_time[self._cooldown_key(severity, source)] = time.time()
            self._dispatch_alert(alert)
            return alert

    def _is_cooldown_passed(self, severity: str, source: str) -> bool:
        """检查冷却期是否已过。

        :param severity: 严重级别
        :type severity: str
        :param source: 告警来源
        :type source: str
        :return: 是否已过冷却期
        :rtype: bool
        """
        key = self._cooldown_key(severity, source)
        last = self._last_alert_time.get(key, 0.0)
        cooldown = self.COOLDOWN_SECONDS.get(severity, 300.0)
        return (time.time() - last) >= cooldown

    def _cooldown_key(self, severity: str, source: str) -> str:
        """生成冷却期字典键。

        :param severity: 严重级别
        :type severity: str
        :param source: 告警来源
        :type source: str
        :return: 组合键
        :rtype: str
        """
        return f"{severity}:{source}"

    def _create_alert(self, severity: str, source: str, message: str) -> _Alert:
        """创建告警对象。

        :param severity: 严重级别
        :type severity: str
        :param source: 告警来源
        :type source: str
        :param message: 告警消息
        :type message: str
        :return: 告警对象
        :rtype: _Alert
        """
        alert_id = f"{source}_{int(time.time() * 1000)}"
        return Alert(alert_id=alert_id, severity=severity, source=source, message=message)

    def _dispatch_alert(self, alert: _Alert) -> None:
        """分发告警到所有处理器。

        :param alert: 告警对象
        :type alert: _Alert
        :return: 无
        :rtype: None
        """
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("AlertManager._dispatch_alert: handler error: %s", e)

    def _log_handler(self, alert: _Alert) -> None:
        """日志通道处理器。

        :param alert: 告警对象
        :type alert: _Alert
        :return: 无
        :rtype: None
        """
        log_msg = f"[ALERT {alert.severity.upper()}] {alert.source}: {alert.message}"
        if alert.severity == 'critical':
            logger.critical(log_msg)
        elif alert.severity == 'warning':
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def _file_handler(self, alert: _Alert) -> None:
        """文件通道处理器。

        :param alert: 告警对象
        :type alert: _Alert
        :return: 无
        :rtype: None
        """
        if not self.alert_log_path:
            return
        try:
            with open(self.alert_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error("AlertManager._file_handler: write failed: %s", e)

    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取活跃告警列表。

        :param severity_filter: 严重级别过滤
        :type severity_filter: Optional[str]
        :return: 告警字典列表
        :rtype: List[Dict[str, Any]]
        """
        with self._lock:
            alerts = [a for a in self._alerts if not a.acknowledged]
            if severity_filter:
                alerts = [a for a in alerts if a.severity == severity_filter]
            return [a.to_dict() for a in alerts]

    def acknowledge(self, alert_id: str) -> bool:
        """确认告警。

        :param alert_id: 告警ID
        :type alert_id: str
        :return: 是否成功确认
        :rtype: bool
        """
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
            return False

    def clear_old_alerts(self, max_age_seconds: float = 86400.0) -> int:
        """清理过期告警。

        :param max_age_seconds: 最大保留时间
        :type max_age_seconds: float
        :return: 清理数量
        :rtype: int
        """
        with self._lock:
            now = time.time()
            before = len(self._alerts)
            self._alerts = [a for a in self._alerts if (now - a.timestamp) < max_age_seconds]
            return before - len(self._alerts)

    def get_stats(self) -> Dict[str, Any]:
        """获取告警统计。

        :param: 无
        :type: 无
        :return: 统计字典
        :rtype: Dict[str, Any]
        """
        with self._lock:
            active = [a for a in self._alerts if not a.acknowledged]
            return {
                'total_alerts': len(self._alerts),
                'active_alerts': len(active),
                'critical_count': len([a for a in active if a.severity == 'critical']),
                'warning_count': len([a for a in active if a.severity == 'warning']),
                'info_count': len([a for a in active if a.severity == 'info']),
            }


class AlertManager(_AlertManager):
    """AlertManager 实现"""

    def trigger(self, severity: str, source: str, message: str) -> Optional[_Alert]:
        """触发告警，受冷却期控制。

        :param severity: 严重级别
        :type severity: str
        :param source: 告警来源模块
        :type source: str
        :param message: 告警消息
        :type message: str
        :return: 告警对象或 None
        :rtype: Optional[_Alert]
        """
        return self._trigger_impl(severity, source, message)

    def _trigger_impl(self, severity: str, source: str, message: str) -> Optional[_Alert]:
        return super().trigger(severity, source, message)
