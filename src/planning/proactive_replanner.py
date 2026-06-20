"""主动式重规划与退化检测。

在长任务执行过程中监控性能退化趋势（延迟增长、成功率下降、连续失败），
在硬失败发生前主动触发重规划，避免任务中断。

第一性原理：
- 等待失败再重规划是反应式的；自进化 Agent 应识别退化前兆并提前干预
- 退化信号：延迟持续增长、成功率持续下降、连续失败累积
- 滑动窗口捕获近期趋势，避免被历史数据稀释
- 冷却期防止退化触发后频繁重规划导致抖动
"""

from collections import deque
from typing import Any, Callable, Dict, List, Optional

from src.task.task_definition import Task


class DegradationDetector:
    """检测执行过程中的性能退化趋势。

    使用滑动窗口监控三类信号：
    - 延迟增长率：近期平均延迟 / 窗口前半段平均延迟
    - 成功率：近期成功率低于阈值
    - 连续失败数：连续失败次数超过阈值

    :param window_size: 滑动窗口大小（最近 N 条记录）
    :param latency_growth_threshold: 延迟增长率阈值，超过则视为退化
    :param success_rate_threshold: 成功率阈值，低于则视为退化
    :param consecutive_failure_threshold: 连续失败次数阈值
    """

    def __init__(
        self,
        window_size: int = 6,
        latency_growth_threshold: float = 2.0,
        success_rate_threshold: float = 0.5,
        consecutive_failure_threshold: int = 3,
    ) -> None:
        self.window_size = max(4, window_size)
        self.latency_growth_threshold = max(1.0, latency_growth_threshold)
        self.success_rate_threshold = max(0.0, min(1.0, success_rate_threshold))
        self.consecutive_failure_threshold = max(1, consecutive_failure_threshold)
        self._window: deque = deque(maxlen=self.window_size)
        self._consecutive_failures = 0

    def observe(self, record: Dict[str, Any]) -> None:
        """观察一条执行记录。"""
        self._window.append({
            'success': record.get('success', False),
            'duration': record.get('duration', 0.0),
        })
        if record.get('success', False):
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

    def is_degraded(self) -> bool:
        """检测当前是否处于退化状态。"""
        return self._check_latency_growth() or self._check_success_rate() or self._check_consecutive_failures()

    def get_signals(self) -> Dict[str, Any]:
        """返回退化信号详情。"""
        records = list(self._window)
        return {
            'window_size': len(records),
            'consecutive_failures': self._consecutive_failures,
            'latency_growth': self._compute_latency_growth(),
            'recent_success_rate': self._compute_success_rate(),
            'is_degraded': self.is_degraded(),
            'reasons': self._degradation_reasons(),
        }

    def reset(self) -> None:
        """清空观察数据。"""
        self._window.clear()
        self._consecutive_failures = 0

    def _compute_latency_growth(self) -> float:
        """计算延迟增长率：后半段均值 / 前半段均值。"""
        records = list(self._window)
        if len(records) < self.window_size:
            return 1.0
        mid = len(records) // 2
        first_half = records[:mid]
        second_half = records[mid:]
        first_avg = sum(r['duration'] for r in first_half) / len(first_half) if first_half else 0
        second_avg = sum(r['duration'] for r in second_half) / len(second_half) if second_half else 0
        if first_avg <= 0:
            return 1.0 if second_avg <= 0 else float('inf')
        return second_avg / first_avg

    def _compute_success_rate(self) -> float:
        """计算窗口内成功率。"""
        records = list(self._window)
        if not records:
            return 1.0
        successes = sum(1 for r in records if r['success'])
        return successes / len(records)

    def _check_latency_growth(self) -> bool:
        return self._compute_latency_growth() > self.latency_growth_threshold

    def _check_success_rate(self) -> bool:
        records = list(self._window)
        if len(records) < self.window_size:
            return False
        return self._compute_success_rate() < self.success_rate_threshold

    def _check_consecutive_failures(self) -> bool:
        return self._consecutive_failures >= self.consecutive_failure_threshold

    def _degradation_reasons(self) -> List[str]:
        reasons: List[str] = []
        if self._check_latency_growth():
            reasons.append('latency_growth')
        if self._check_success_rate():
            reasons.append('low_success_rate')
        if self._check_consecutive_failures():
            reasons.append('consecutive_failures')
        return reasons


class ProactiveReplanner:
    """主动式重规划器：在退化检测触发时主动重规划。

    :param detector: 退化检测器
    :param replan_callback: 重规划回调，签名为
        ``callback(planner, current_index, signals) -> Optional[List[Task]]``
    :param cooldown: 重规划冷却期（步数），防止频繁触发抖动
    """

    def __init__(
        self,
        detector: DegradationDetector,
        replan_callback: Callable[..., Optional[List[Task]]],
        cooldown: int = 3,
    ) -> None:
        self.detector = detector
        self.replan_callback = replan_callback
        self.cooldown = max(1, cooldown)
        self._steps_since_replan = 0
        self._replan_count = 0

    @property
    def replan_count(self) -> int:
        """返回已触发的重规划次数。"""
        return self._replan_count

    def observe(self, record: Dict[str, Any]) -> None:
        """观察执行记录并更新检测器。"""
        self.detector.observe(record)
        self._steps_since_replan += 1

    def check(
        self,
        planner: Any,
        current_index: int,
    ) -> Optional[List[Task]]:
        """检查是否需要主动重规划。

        :param planner: 长任务规划器实例
        :param current_index: 当前步骤索引
        :return: 新的子任务列表，或 None 表示无需重规划
        """
        if self._steps_since_replan < self.cooldown:
            return None
        if not self.detector.is_degraded():
            return None

        signals = self.detector.get_signals()
        new_steps = self.replan_callback(planner, current_index, signals)
        if new_steps:
            self._replan_count += 1
            self._steps_since_replan = 0
            self.detector.reset()
        return new_steps

    def reset(self) -> None:
        """重置规划器状态。"""
        self.detector.reset()
        self._steps_since_replan = 0
        self._replan_count = 0
