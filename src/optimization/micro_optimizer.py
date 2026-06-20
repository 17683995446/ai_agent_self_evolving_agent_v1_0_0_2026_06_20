from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from src.prediction.base_predictor import AdaptiveExponentialSmoother
    from src.prediction.multi_predictor import MultiPredictor


class BaseMicroOptimizer(abc.ABC):
    def __init__(self, learning_rate: float = 0.01, min_improvement: float = 0.001) -> None:
        """初始化微优化器。

        :param learning_rate: 学习率
        :type learning_rate: float
        :param min_improvement: 最小改进阈值
        :type min_improvement: float
        :return: 无
        :rtype: None
        """
        self.learning_rate = learning_rate
        self.min_improvement = min_improvement
        self._error_history: Dict[str, List[float]] = {}
        self._param_history: Dict[str, List[Dict[str, Any]]] = {}

    @abc.abstractmethod
    def record_errors(self, errors: Dict[str, float]) -> None:
        """Record error history without adjusting parameters.

        :param errors: 各指标的错误值字典
        :type errors: Dict[str, float]
        :return: 无
        :rtype: None
        """

    @abc.abstractmethod
    def optimize(self, predictor: Optional[MultiPredictor], errors: Dict[str, float]) -> Dict[str, Any]:
        """Optimize predictor parameters based on errors.

        :param predictor: 多指标预测器实例
        :type predictor: Optional[MultiPredictor]
        :param errors: 各指标的错误值字典
        :type errors: Dict[str, float]
        :return: 优化改进结果字典
        :rtype: Dict[str, Any]
        """

    @abc.abstractmethod
    def optimize_adaptive_smoother(self, smoother: AdaptiveExponentialSmoother,
                                    actual: float) -> None:
        """Optimize an adaptive exponential smoother.

        :param smoother: 自适应指数平滑器实例
        :type smoother: AdaptiveExponentialSmoother
        :param actual: 实际观测值
        :type actual: float
        :return: 无
        :rtype: None
        """

    @abc.abstractmethod
    def reset_metric_history(self, metric: str) -> None:
        """Reset history for a metric.

        :param metric: 指标名称
        :type metric: str
        :return: 无
        :rtype: None
        """

    @abc.abstractmethod
    def get_metric_trend(self, metric: str) -> Optional[float]:
        """Return the slope of recent errors for a metric.

        :param metric: 指标名称
        :type metric: str
        :return: 误差趋势斜率，数据不足时返回None
        :rtype: Optional[float]
        """


class MicroOptimizer(BaseMicroOptimizer):
    def record_errors(self, errors: Dict[str, float]) -> None:
        """Record error history without adjusting parameters.

        :param errors: 各指标的错误值字典
        :type errors: Dict[str, float]
        :return: 无
        :rtype: None
        """
        for metric, error in errors.items():
            if metric not in self._error_history:
                self._error_history[metric] = []
            self._error_history[metric].append(error)

    def optimize(self, predictor: Optional[MultiPredictor], errors: Dict[str, float]) -> Dict[str, Any]:
        """Optimize predictor parameters based on errors.

        :param predictor: 多指标预测器实例
        :type predictor: Optional[MultiPredictor]
        :param errors: 各指标的错误值字典
        :type errors: Dict[str, float]
        :return: 优化改进结果字典
        :rtype: Dict[str, Any]
        """
        improvements: Dict[str, Any] = {}
        for metric, error in errors.items():
            result = self._optimize_metric(predictor, metric, error)
            if result:
                improvements[metric] = result
        return improvements

    def _optimize_metric(self, predictor: Optional[MultiPredictor],
                         metric: str, error: float) -> Optional[Dict[str, Any]]:
        self._record_error(metric, error)
        improvement = self._compute_improvement(metric, error)
        if improvement is None or predictor is None:
            return None
        if improvement < self.min_improvement:
            return self._try_adjust_metric(predictor, metric, error, improvement)
        return None

    def _record_error(self, metric: str, error: float) -> None:
        if metric not in self._error_history:
            self._error_history[metric] = []
        self._error_history[metric].append(error)

    def _compute_improvement(self, metric: str, error: float) -> Optional[float]:
        if len(self._error_history[metric]) < 2:
            return None
        prev_error = self._error_history[metric][-2]
        return prev_error - error

    def _try_adjust_metric(self, predictor: MultiPredictor, metric: str,
                           error: float, improvement: float) -> Optional[Dict[str, Any]]:
        params = predictor.get_metric_params(metric)
        if params is None:
            return None
        new_params = self._adjust_params(params, error, self._error_history[metric][-2])
        predictor.set_metric_params(metric, new_params)
        return self._build_adjust_result(params, new_params, improvement)

    def _build_adjust_result(self, params: Dict[str, Any], new_params: Dict[str, Any],
                             improvement: float) -> Dict[str, Any]:
        return {
            'old_params': params,
            'new_params': new_params,
            'improvement': improvement,
        }

    def _adjust_params(self, params: Dict[str, Any], current_error: float,
                       prev_error: float) -> Dict[str, Any]:
        """Adjust parameters based on error trend."""
        new_params = dict(params)
        if 'alpha' in params and 'level' in params:
            gradient = (current_error - prev_error) / max(prev_error, 1e-10)
            new_alpha = params['alpha'] - self.learning_rate * gradient
            new_params['alpha'] = max(0.05, min(0.8, new_alpha))
        return new_params

    def optimize_adaptive_smoother(self, smoother: AdaptiveExponentialSmoother,
                                    actual: float) -> None:
        """Optimize an adaptive exponential smoother.

        修复Bug6a: 使用 `is not None` 判断 level，避免 level=0.0 被误判为 falsy。
        修复Bug6b: 更新 smoother.level 和 smoother.last_error，使平滑器能够学习。

        :param smoother: 自适应指数平滑器实例
        :type smoother: AdaptiveExponentialSmoother
        :param actual: 实际观测值
        :type actual: float
        :return: 无
        :rtype: None
        """
        if smoother.last_error is None:
            # 首次调用，初始化level
            smoother.level = actual
            smoother.last_error = 0.0
            return
        # 修复Bug6a: 用 `is not None` 替代 `if smoother.level`，避免0.0误判
        if smoother.level is not None:
            current_error = abs(actual - smoother.level)
        else:
            current_error = 0.0
        self._update_smoother_alpha(smoother, current_error)
        # 修复Bug6b: 更新 level 和 last_error，使平滑器能够学习
        alpha = smoother.alpha
        smoother.level = alpha * actual + (1 - alpha) * smoother.level
        smoother.last_error = current_error

    def _update_smoother_alpha(self, smoother: AdaptiveExponentialSmoother,
                               current_error: float) -> None:
        prev_error = smoother.last_error
        if current_error > prev_error:
            smoother.alpha = min(smoother.alpha_max,
                                  smoother.alpha + smoother.adaptation_rate * 2)
        else:
            smoother.alpha = max(smoother.alpha_min,
                                  smoother.alpha - smoother.adaptation_rate)

    def reset_metric_history(self, metric: str) -> None:
        """Reset history for a metric.

        :param metric: 指标名称
        :type metric: str
        :return: 无
        :rtype: None
        """
        if metric in self._error_history:
            self._error_history[metric] = []
        if metric in self._param_history:
            self._param_history[metric] = []

    def get_metric_trend(self, metric: str) -> Optional[float]:
        """Return the slope of recent errors for a metric.

        :param metric: 指标名称
        :type metric: str
        :return: 误差趋势斜率，数据不足时返回None
        :rtype: Optional[float]
        """
        errors = self._get_recent_errors(metric)
        if errors is None:
            return None
        return self._compute_slope(errors)

    def _get_recent_errors(self, metric: str) -> Optional[List[float]]:
        if metric not in self._error_history or len(self._error_history[metric]) < 2:
            return None
        errors = self._error_history[metric][-10:]
        if len(errors) < 2:
            return None
        return errors

    def _compute_slope(self, errors: List[float]) -> float:
        x = np.arange(len(errors))
        slope = np.polyfit(x, errors, 1)[0]
        return float(slope)
