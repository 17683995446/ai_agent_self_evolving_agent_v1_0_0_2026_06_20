import abc
import logging
from typing import Dict, Any, Optional, Callable

from .snapshot_manager import SnapshotManager

logger = logging.getLogger(__name__)


class BaseAutoRollback(abc.ABC):
    def __init__(self, snapshot_manager: SnapshotManager,
                 mae_degradation_threshold: float = 0.2) -> None:
        """Initialize the auto rollback manager.

        :param snapshot_manager: Snapshot manager instance.
        :type snapshot_manager: SnapshotManager
        :param mae_degradation_threshold: MAE degradation threshold.
        :type mae_degradation_threshold: float
        :return: None
        :rtype: None
        """
        self.snapshot_manager = snapshot_manager
        self.mae_degradation_threshold = self._sanitize_threshold(mae_degradation_threshold)
        self._baseline_mae: Optional[float] = None
        self._baseline_snapshot_id: Optional[str] = None
        self._rollback_count: int = 0
        # 修复Bug5: 添加退化检测窗口，需连续3个周期退化才触发回滚
        self._degradation_streak: int = 0
        self._degradation_window: int = 3

    @staticmethod
    def _sanitize_threshold(value: Any) -> float:
        """校正 mae_degradation_threshold 参数，确保为非负浮点数。

        :param value: 原始值。
        :type value: Any
        :return: 校正后的非负浮点数。
        :rtype: float
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.2
        return v if v >= 0 else 0.2

    @abc.abstractmethod
    def set_baseline(self, mae: float, snapshot_id: str) -> None:
        """Set the baseline MAE and snapshot ID.

        :param mae: Baseline MAE value.
        :type mae: float
        :param snapshot_id: Baseline snapshot ID.
        :type snapshot_id: str
        :return: None
        :rtype: None
        """

    @abc.abstractmethod
    def check_and_rollback(self, current_mae: float,
                           restore_callback: Optional[Callable[[Any], bool]] = None) -> Dict[str, Any]:
        """Check if rollback is needed and perform it.

        :param current_mae: Current MAE value.
        :type current_mae: float
        :param restore_callback: Optional callback to restore state.
        :type restore_callback: Optional[Callable[[Any], bool]]
        :return: Rollback result dictionary.
        :rtype: Dict[str, Any]
        """

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return rollback statistics.

        :param: 无
        :type: 无
        :return: Statistics dictionary.
        :rtype: Dict[str, Any]
        """

    def _is_degraded(self, current_mae: float) -> bool:
        """Check if current MAE indicates degradation.

        :param current_mae: Current MAE value.
        :type current_mae: float
        :return: Whether degraded.
        :rtype: bool
        """
        if self._baseline_mae is None:
            return False
        return (current_mae - self._baseline_mae) >= (self.mae_degradation_threshold - 1e-9)

    def get_rollback_count(self) -> int:
        """Return the number of rollbacks performed.

        :param: 无
        :type: 无
        :return: Rollback count.
        :rtype: int
        """
        return self._rollback_count


class AutoRollback(BaseAutoRollback):
    def __init__(self, snapshot_manager: SnapshotManager,
                 mae_degradation_threshold: float = 0.2,
                 on_rollback: Optional[Callable[[str], None]] = None) -> None:
        """Initialize the auto rollback manager.

        :param snapshot_manager: Snapshot manager instance.
        :type snapshot_manager: SnapshotManager
        :param mae_degradation_threshold: MAE degradation threshold.
        :type mae_degradation_threshold: float
        :param on_rollback: Optional callback on rollback.
        :type on_rollback: Optional[Callable[[str], None]]
        :return: None
        :rtype: None
        """
        super().__init__(snapshot_manager, mae_degradation_threshold)
        self._on_rollback = on_rollback

    def set_baseline(self, mae: float, snapshot_id: str) -> None:
        """Set the baseline MAE and snapshot ID.

        :param mae: Baseline MAE value.
        :type mae: float
        :param snapshot_id: Baseline snapshot ID.
        :type snapshot_id: str
        :return: None
        :rtype: None
        """
        self._baseline_mae = mae
        self._baseline_snapshot_id = snapshot_id
        logger.info("AutoRollback.set_baseline: mae=%.4f snapshot_id=%s", mae, snapshot_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return rollback statistics.

        :param: 无
        :type: 无
        :return: Statistics dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'baseline_mae': self._baseline_mae,
            'baseline_snapshot_id': self._baseline_snapshot_id,
            'rollback_count': self._rollback_count,
            'threshold': self.mae_degradation_threshold,
        }

    def check_and_rollback(self, current_mae: float,
                           restore_callback: Optional[Callable[[Any], bool]] = None) -> Dict[str, Any]:
        """Check if rollback is needed and perform it.

        修复Bug5: 需连续_degradation_window个周期退化才触发回滚，避免单次毛刺误触发。

        :param current_mae: Current MAE value.
        :type current_mae: float
        :param restore_callback: Optional callback to restore state.
        :type restore_callback: Optional[Callable[[Any], bool]]
        :return: Rollback result dictionary.
        :rtype: Dict[str, Any]
        """
        if self._baseline_mae is None:
            logger.debug("AutoRollback.check_and_rollback: no baseline set")
            return {'action': 'no_baseline'}
        if self._is_degraded(current_mae):
            self._degradation_streak += 1
            logger.warning("AutoRollback.check_and_rollback: degradation detected, streak=%d/%d",
                           self._degradation_streak, self._degradation_window)
            if self._degradation_streak < self._degradation_window:
                return {'action': 'continue', 'degradation_streak': self._degradation_streak}
        else:
            # 退化恢复，重置计数
            if self._degradation_streak > 0:
                logger.info("AutoRollback.check_and_rollback: degradation cleared, streak reset")
                self._degradation_streak = 0
            return {'action': 'continue'}
        # 连续退化达到窗口阈值，触发回滚
        if self._baseline_snapshot_id is None:
            logger.error("AutoRollback.check_and_rollback: rollback triggered but baseline snapshot missing")
            return {'action': 'rollback_failed', 'reason': 'snapshot_not_found'}
        # 重置退化计数，避免反复回滚
        logger.critical("AutoRollback.check_and_rollback: threshold reached, rolling back to snapshot=%s",
                        self._baseline_snapshot_id)
        self._degradation_streak = 0
        return self._perform_rollback(restore_callback)

    def _perform_rollback(self, restore_callback: Optional[Callable[[Any], bool]] = None) -> Dict[str, Any]:
        """Perform the rollback operation.

        :param restore_callback: Optional callback to restore state.
        :type restore_callback: Optional[Callable[[Any], bool]]
        :return: Rollback result dictionary.
        :rtype: Dict[str, Any]
        """
        try:
            state = self.snapshot_manager.load_snapshot(self._baseline_snapshot_id)
            if state is None:
                logger.error("AutoRollback._perform_rollback: snapshot not found id=%s", self._baseline_snapshot_id)
                return {'action': 'rollback_failed', 'reason': 'snapshot_not_found'}
            if restore_callback is not None:
                success = restore_callback(state)
                if not success:
                    logger.error("AutoRollback._perform_rollback: restore callback failed")
                    return {'action': 'rollback_failed', 'reason': 'callback_failed'}
            self._rollback_count += 1
            logger.critical("AutoRollback._perform_rollback: rollback completed, count=%d", self._rollback_count)
            if self._on_rollback:
                self._on_rollback(self._baseline_snapshot_id)
            return {'action': 'rollback', 'success': True}
        except Exception as e:
            logger.exception("AutoRollback._perform_rollback: rollback failed: %s", e)
            return {'action': 'rollback_failed', 'reason': 'snapshot_not_found'}
