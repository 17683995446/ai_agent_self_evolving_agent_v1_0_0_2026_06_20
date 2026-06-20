"""全链路运行时追踪器：为每个模块提供详细的执行日志"""

import abc
import time
import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class BaseCycleTracer(abc.ABC):
    """周期级追踪器抽象基类"""

    def __init__(self) -> None:
        """初始化追踪器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        from collections import deque
        self._trace_logs: deque = deque(maxlen=5000)
        self._current_cycle = 0
        self._enabled = True

    @abc.abstractmethod
    def enable(self) -> None:
        """启用追踪器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def disable(self) -> None:
        """禁用追踪器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def trace(self, phase: str, message: str, data: Optional[Dict] = None) -> None:
        """记录追踪点。

        :param phase: 阶段名称。
        :type phase: str
        :param message: 追踪消息。
        :type message: str
        :param data: 附加数据。
        :type data: Optional[Dict]
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def set_cycle(self, cycle: int) -> None:
        """设置当前周期。

        :param cycle: 周期编号。
        :type cycle: int
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def get_recent_traces(self, count: int = 100) -> list:
        """获取最近的追踪记录。

        :param count: 返回条数。
        :type count: int
        :return: 追踪记录列表。
        :rtype: list
        """
        ...

    @abc.abstractmethod
    def get_traces_by_phase(self, phase: str) -> list:
        """按阶段获取追踪记录。

        :param phase: 阶段名称。
        :type phase: str
        :return: 追踪记录列表。
        :rtype: list
        """
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """清空追踪记录。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        ...

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取追踪统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """
        ...


class CycleTracer(BaseCycleTracer):
    """周期级追踪器：记录每个周期的完整执行链路"""

    def __init__(self) -> None:
        """初始化追踪器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        super().__init__()

    def enable(self) -> None:
        """启用追踪器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._enabled = True

    def disable(self) -> None:
        """禁用追踪器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._enabled = False

    def trace(self, phase: str, message: str, data: Optional[Dict] = None) -> None:
        """记录追踪点。

        :param phase: 阶段名称。
        :type phase: str
        :param message: 追踪消息。
        :type message: str
        :param data: 附加数据。
        :type data: Optional[Dict]
        :return: 无。
        :rtype: None
        """
        if not self._enabled:
            return
        self._add_trace_entry(phase, message, data)
        self._log_trace(phase, message, data)

    def _add_trace_entry(self, phase: str, message: str, data: Optional[Dict]) -> None:
        entry = {
            'cycle': self._current_cycle,
            'phase': phase,
            'message': message,
            'timestamp': time.time(),
            'data': data or {},
        }
        self._trace_logs.append(entry)

    def _log_trace(self, phase: str, message: str, data: Optional[Dict]) -> None:
        log_msg = f"[CYCLE-{self._current_cycle}] [{phase}] {message}"
        if data:
            log_msg += f" | data={data}"
        logger.debug(log_msg)

    @contextmanager
    def trace_block(self, phase: str, label: str):
        """追踪代码块执行时间。

        :param phase: 阶段名称。
        :type phase: str
        :param label: 代码块标签。
        :type label: str
        :return: 上下文管理器。
        :rtype: contextmanager
        """
        start = time.time()
        self.trace(phase, f"START {label}")
        try:
            yield
        except Exception as e:
            self.trace(phase, f"ERROR {label}: {e}", {'error': str(e)})
            raise
        finally:
            duration = time.time() - start
            self.trace(phase, f"END {label}", {'duration_ms': round(duration * 1000, 2)})

    def set_cycle(self, cycle: int) -> None:
        """设置当前周期。

        :param cycle: 周期编号。
        :type cycle: int
        :return: 无。
        :rtype: None
        """
        self._current_cycle = cycle

    def get_recent_traces(self, count: int = 100) -> list:
        """获取最近的追踪记录。

        :param count: 返回条数。
        :type count: int
        :return: 追踪记录列表。
        :rtype: list
        """
        return list(self._trace_logs)[-count:]

    def get_traces_by_phase(self, phase: str) -> list:
        """按阶段获取追踪记录。

        :param phase: 阶段名称。
        :type phase: str
        :return: 追踪记录列表。
        :rtype: list
        """
        return [t for t in self._trace_logs if t['phase'] == phase]

    def clear(self) -> None:
        """清空追踪记录。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._trace_logs.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取追踪统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """
        return {
            'total_traces': len(self._trace_logs),
            'current_cycle': self._current_cycle,
            'phases': self._count_phases(),
        }

    def _count_phases(self) -> Dict[str, int]:
        phases = {}
        for t in self._trace_logs:
            p = t['phase']
            phases[p] = phases.get(p, 0) + 1
        return phases


def get_tracer() -> CycleTracer:
    """获取全局追踪器实例。

    :param: 无
    :type: 无
    :return: 追踪器实例。
    :rtype: CycleTracer
    """
    return CycleTracer()
