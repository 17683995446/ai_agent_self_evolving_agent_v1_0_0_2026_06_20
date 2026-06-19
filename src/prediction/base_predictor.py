import abc
import math
from typing import List, Optional, Dict, Any
from collections import deque


class BasePredictor(abc.ABC):
    """Abstract base class for all predictors."""

    @abc.abstractmethod
    def predict(self, steps: int = 1) -> List[float]:
        """Predict future values.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Predicted future values.
        :rtype: List[float]
        """

    @abc.abstractmethod
    def update(self, actual: float) -> None:
        """Update predictor with actual value.

        :param actual: Actual observed value.
        :type actual: float
        :return: None
        :rtype: None
        """


class ExponentialSmoother(BasePredictor):
    """Exponential smoothing predictor."""

    def __init__(self, alpha: float = 0.3) -> None:
        """Initialize the exponential smoother.

        :param alpha: Smoothing factor.
        :type alpha: float
        :return: None
        :rtype: None
        """
        self.alpha = alpha
        self.level: Optional[float] = None
        self._history: deque = deque(maxlen=1000)

    def fit(self, series: List[float]) -> None:
        """Fit model to series.

        :param series: Time series data to fit.
        :type series: List[float]
        :return: None
        :rtype: None
        """
        if len(series) == 0:
            return
        self.level = float(series[0])
        self._history.clear()
        self._history.extend(series)
        for value in series[1:]:
            self.level = self.alpha * value + (1 - self.alpha) * self.level

    def predict(self, steps: int = 1) -> List[float]:
        """Predict future values.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Predicted future values.
        :rtype: List[float]
        """
        if self.level is None:
            return [0.0] * steps
        return [self.level] * steps

    def update(self, actual: float) -> None:
        """Update predictor with actual value.

        :param actual: Actual observed value.
        :type actual: float
        :return: None
        :rtype: None
        """
        # 修复阶段123: 过滤None/NaN/Inf，避免污染预测器状态
        if actual is None or math.isnan(actual) or math.isinf(actual):
            return
        self._history.append(actual)
        if self.level is None:
            self.level = float(actual)
        else:
            self.level = self.alpha * actual + (1 - self.alpha) * self.level

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters.

        :param: 无
        :type: 无
        :return: Current parameters dictionary.
        :rtype: Dict[str, Any]
        """
        return {'alpha': self.alpha, 'level': self.level}

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters from dict.

        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.alpha = params.get('alpha', self.alpha)
        self.level = params.get('level', self.level)


class AdaptiveExponentialSmoother(BasePredictor):
    """Adaptive exponential smoothing predictor."""

    def __init__(self, alpha_init: float = 0.3, alpha_min: float = 0.05,
                 alpha_max: float = 0.8, adaptation_rate: float = 0.01) -> None:
        """Initialize the adaptive exponential smoother.

        :param alpha_init: Initial smoothing factor.
        :type alpha_init: float
        :param alpha_min: Minimum smoothing factor.
        :type alpha_min: float
        :param alpha_max: Maximum smoothing factor.
        :type alpha_max: float
        :param adaptation_rate: Rate of alpha adaptation.
        :type adaptation_rate: float
        :return: None
        :rtype: None
        """
        self.alpha = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.adaptation_rate = adaptation_rate
        self.level: Optional[float] = None
        self.last_error: Optional[float] = None
        self._history: deque = deque(maxlen=1000)

    def predict(self, steps: int = 1) -> List[float]:
        """Predict future values.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Predicted future values.
        :rtype: List[float]
        """
        if self.level is None:
            return [0.0] * steps
        return [self.level] * steps

    def _adapt_alpha(self, error: float) -> None:
        """Adapt alpha based on error change.

        :param error: Prediction error.
        :type error: float
        :return: None
        :rtype: None
        """
        if self.last_error is None:
            return
        if error > self.last_error:
            self.alpha = min(self.alpha_max, self.alpha + self.adaptation_rate)
        else:
            self.alpha = max(self.alpha_min, self.alpha - self.adaptation_rate)

    def update(self, actual: float) -> None:
        """Update predictor with actual value.

        :param actual: Actual observed value.
        :type actual: float
        :return: None
        :rtype: None
        """
        # 修复阶段123: 过滤None/NaN/Inf，避免污染预测器状态
        if actual is None or math.isnan(actual) or math.isinf(actual):
            return
        self._history.append(actual)
        if self.level is None:
            self._init_level(actual)
            return
        self._update_level(actual)

    def _init_level(self, actual: float) -> None:
        """Initialize level and last_error.

        :param actual: Actual observed value.
        :type actual: float
        :return: None
        :rtype: None
        """
        self.level = float(actual)
        self.last_error = abs(actual - self.level)

    def _update_level(self, actual: float) -> None:
        """Update level with new actual value.

        :param actual: Actual observed value.
        :type actual: float
        :return: None
        :rtype: None
        """
        prediction = self.level
        error = abs(actual - prediction)
        self._adapt_alpha(error)
        self.level = self.alpha * actual + (1 - self.alpha) * self.level
        self.last_error = error

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters.

        :param: 无
        :type: 无
        :return: Current parameters dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'alpha': self.alpha,
            'level': self.level,
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters from dict.

        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.alpha = params.get('alpha', self.alpha)
        self.level = params.get('level', self.level)
        self.alpha_min = params.get('alpha_min', self.alpha_min)
        self.alpha_max = params.get('alpha_max', self.alpha_max)


class SimpleMovingAverage(BasePredictor):
    """Simple moving average predictor."""

    def __init__(self, window: int = 10) -> None:
        """Initialize the simple moving average predictor.

        :param window: Window size for moving average.
        :type window: int
        :return: None
        :rtype: None
        """
        self.window = window
        self._history: deque = deque(maxlen=window)

    def predict(self, steps: int = 1) -> List[float]:
        """Predict future values.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Predicted future values.
        :rtype: List[float]
        """
        if len(self._history) == 0:
            return [0.0] * steps
        avg = sum(self._history) / len(self._history)
        return [avg] * steps

    def update(self, actual: float) -> None:
        """Update predictor with actual value.

        :param actual: Actual observed value.
        :type actual: float
        :return: None
        :rtype: None
        """
        # 修复阶段123: 过滤None/NaN/Inf，避免污染预测器状态
        if actual is None or math.isnan(actual) or math.isinf(actual):
            return
        self._history.append(actual)

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters.

        :param: 无
        :type: 无
        :return: Current parameters dictionary.
        :rtype: Dict[str, Any]
        """
        return {'window': self.window, 'history': list(self._history)}

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters from dict.

        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.window = params.get('window', self.window)
        if 'history' in params:
            self._history.clear()
            self._history.extend(params['history'])
