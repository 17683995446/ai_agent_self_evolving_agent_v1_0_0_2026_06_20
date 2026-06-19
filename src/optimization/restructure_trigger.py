import abc
from typing import Dict, Any, List, Optional
from collections import deque


class BaseRestructureTrigger(abc.ABC):
    def __init__(self, high_error_threshold: float = 0.5,
                 stable_window: int = 20,
                 max_restructures: int = 3) -> None:
        """Initialize the restructure trigger.

        :param high_error_threshold: MAE threshold to trigger restructure.
        :type high_error_threshold: float
        :param stable_window: Number of errors considered for stable degradation.
        :type stable_window: int
        :param max_restructures: Maximum allowed restructures.
        :type max_restructures: int
        :return: None
        :rtype: None
        """
        self.high_error_threshold = high_error_threshold
        self.stable_window = stable_window
        self.max_restructures = max_restructures
        self._restructure_count: int = 0
        self._error_history: deque = deque(maxlen=stable_window)
        self._last_restructure_time: Optional[float] = None

    @abc.abstractmethod
    def _abstract_trigger_op(self) -> None:
        """Abstract trigger operation placeholder."""

    def should_restructure(self, current_mae: float,
                           recent_errors: List[float]) -> bool:
        """Determine whether a restructure should be triggered.

        :param current_mae: Current MAE value.
        :type current_mae: float
        :param recent_errors: List of recent error values.
        :type recent_errors: List[float]
        :return: Whether restructure should be triggered.
        :rtype: bool
        """
        if self._restructure_count >= self.max_restructures:
            return False
        if current_mae > self.high_error_threshold:
            return True
        return self._check_stable_degradation(recent_errors)

    def _check_stable_degradation(self, recent_errors: List[float]) -> bool:
        if len(recent_errors) < self.stable_window:
            return False
        recent_avg = sum(recent_errors[-self.stable_window:]) / self.stable_window
        older_avg = self._compute_older_avg(recent_errors, recent_avg)
        return recent_avg >= older_avg * 0.95

    def _compute_older_avg(self, recent_errors: List[float], recent_avg: float) -> float:
        if len(recent_errors) >= self.stable_window * 2:
            return sum(recent_errors[:self.stable_window]) / self.stable_window
        return recent_avg

    def record_restructure(self) -> None:
        """Record that a restructure occurred.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._restructure_count += 1
        import time
        self._last_restructure_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Return restructure statistics.

        :param: 无
        :type: 无
        :return: Dictionary containing restructure statistics.
        :rtype: Dict[str, Any]
        """
        return {
            'restructure_count': self._restructure_count,
            'max_restructures': self.max_restructures,
            'last_restructure_time': self._last_restructure_time,
            'remaining': self.max_restructures - self._restructure_count,
        }

    def reset(self) -> None:
        """Reset restructure state.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._restructure_count = 0
        self._error_history.clear()
        self._last_restructure_time = None


class RestructureTrigger(BaseRestructureTrigger):
    def _abstract_trigger_op(self) -> None:
        """Concrete implementation of abstract trigger operation."""
        pass
