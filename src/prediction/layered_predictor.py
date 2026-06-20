import abc
import time
from collections import deque
from typing import Any, Dict, List, Optional

from .arima_predictor import ARIMAModel
from .base_predictor import AdaptiveExponentialSmoother


class BaseLayerPredictor(abc.ABC):
    """Abstract base class for layered predictors."""

    @abc.abstractmethod
    def predict(self) -> Dict[str, float]:
        """Return predictions.

        :param: 无
        :type: 无
        :return: Predictions dictionary.
        :rtype: Dict[str, float]
        """

    @abc.abstractmethod
    def update(self, actuals: Dict[str, float]) -> None:
        """Update with actual values.

        :param actuals: Actual values dictionary.
        :type actuals: Dict[str, float]
        :return: None
        :rtype: None
        """


class LowLevelPredictor(BaseLayerPredictor):
    """Low-level predictor using adaptive exponential smoothing."""

    def __init__(self, metrics: List[str], window_size: int = 60) -> None:
        """Initialize the low-level predictor.

        :param metrics: List of metric names.
        :type metrics: List[str]
        :param window_size: History window size.
        :type window_size: int
        :return: None
        :rtype: None
        """
        self.metrics = metrics
        self.window_size = window_size
        self.predictors: Dict[str, AdaptiveExponentialSmoother] = {}
        self._history: Dict[str, deque] = {}
        self._init_predictors()

    def _init_predictors(self) -> None:
        """Initialize predictors for each metric.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        for metric in self.metrics:
            self.predictors[metric] = AdaptiveExponentialSmoother(alpha_init=0.5)
            self._history[metric] = deque(maxlen=self.window_size)

    def predict(self) -> Dict[str, float]:
        """Return predictions for all metrics.

        :param: 无
        :type: 无
        :return: Predictions dictionary.
        :rtype: Dict[str, float]
        """
        predictions = {}
        for metric, predictor in self.predictors.items():
            preds = predictor.predict(steps=1)
            predictions[metric] = float(preds[0]) if preds else 0.0
        return predictions

    def update(self, actuals: Dict[str, float]) -> None:
        """Update predictors with actual values.

        :param actuals: Actual values dictionary.
        :type actuals: Dict[str, float]
        :return: None
        :rtype: None
        """
        for metric, value in actuals.items():
            if metric in self.predictors:
                self.predictors[metric].update(float(value))
                self._history[metric].append(float(value))

    def get_history(self, metric: str) -> List[float]:
        """Return history for a metric.

        :param metric: Metric name.
        :type metric: str
        :return: History values.
        :rtype: List[float]
        """
        return list(self._history.get(metric, deque()))

    def get_metric_params(self, metric: str) -> Optional[Dict[str, Any]]:
        """Return parameters for a metric.

        :param metric: Metric name.
        :type metric: str
        :return: Parameters dictionary or None.
        :rtype: Optional[Dict[str, Any]]
        """
        predictor = self.predictors.get(metric)
        if predictor is None:
            return None
        return predictor.get_params()

    def set_metric_params(self, metric: str, params: Dict[str, Any]) -> None:
        """Set parameters for a metric.

        :param metric: Metric name.
        :type metric: str
        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        predictor = self.predictors.get(metric)
        if predictor is not None:
            predictor.set_params(params)


class MidLevelPredictor(BaseLayerPredictor):
    """Mid-level predictor using ARIMA models."""

    def __init__(self, metrics: List[str], trend_window: int = 10) -> None:
        """Initialize the mid-level predictor.

        :param metrics: List of metric names.
        :type metrics: List[str]
        :param trend_window: Trend history window size.
        :type trend_window: int
        :return: None
        :rtype: None
        """
        self.metrics = metrics
        self.trend_window = trend_window
        self.predictors: Dict[str, ARIMAModel] = {}
        self._trend_history: Dict[str, deque] = {}
        self._init_predictors()

    def _init_predictors(self) -> None:
        """Initialize predictors for each metric.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        for metric in self.metrics:
            self.predictors[metric] = ARIMAModel(order=(1, 1, 1), min_samples=5)
            self._trend_history[metric] = deque(maxlen=self.trend_window)

    def predict(self) -> Dict[str, float]:
        """Return predictions for all metrics.

        :param: 无
        :type: 无
        :return: Predictions dictionary.
        :rtype: Dict[str, float]
        """
        predictions = {}
        for metric, predictor in self.predictors.items():
            preds = predictor.predict(steps=1)
            predictions[metric] = float(preds[0]) if preds else 0.0
        return predictions

    def _record_trend(self, metric: str, value: float) -> None:
        """Record trend for a metric.

        :param metric: Metric name.
        :type metric: str
        :param value: Actual value.
        :type value: float
        :return: None
        :rtype: None
        """
        history = self.predictors[metric]._history
        if len(history) >= 2:
            trend = float(value) - list(history)[-2]
            self._trend_history[metric].append(trend)

    def update(self, actuals: Dict[str, float]) -> None:
        """Update predictors with actual values.

        :param actuals: Actual values dictionary.
        :type actuals: Dict[str, float]
        :return: None
        :rtype: None
        """
        for metric, value in actuals.items():
            if metric in self.predictors:
                self.predictors[metric].update(float(value))
                self._record_trend(metric, value)

    def get_trend(self, metric: str) -> Optional[float]:
        """Return average trend for a metric.

        :param metric: Metric name.
        :type metric: str
        :return: Average trend or None.
        :rtype: Optional[float]
        """
        trends = list(self._trend_history.get(metric, deque()))
        if not trends:
            return None
        return sum(trends) / len(trends)


class BaseHighLevelPredictor(abc.ABC):
    """Abstract base class for high-level predictors."""

    @abc.abstractmethod
    def predict(self, low_predictions: Dict[str, float],
                mid_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Return high-level prediction result.

        :param low_predictions: Low-level predictions.
        :type low_predictions: Dict[str, float]
        :param mid_predictions: Mid-level predictions.
        :type mid_predictions: Dict[str, float]
        :return: High-level prediction result.
        :rtype: Dict[str, Any]
        """

    @abc.abstractmethod
    def update(self, state: Dict[str, Any], action_taken: str,
               outcome: float) -> None:
        """Update with state and outcome.

        :param state: Current state.
        :type state: Dict[str, Any]
        :param action_taken: Action taken.
        :type action_taken: str
        :param outcome: Outcome value.
        :type outcome: float
        :return: None
        :rtype: None
        """


class HighLevelPredictor(BaseHighLevelPredictor):
    """High-level predictor combining low and mid predictions."""

    def __init__(self) -> None:
        """Initialize the high-level predictor.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._patterns: Dict[str, Any] = {}
        self._abstract_states: deque = deque(maxlen=100)
        self._strategy_scores: Dict[str, float] = {}

    def _combine(self, low_predictions: Dict[str, float],
                 mid_predictions: Dict[str, float]) -> Dict[str, float]:
        """Combine low and mid predictions.

        :param low_predictions: Low-level predictions.
        :type low_predictions: Dict[str, float]
        :param mid_predictions: Mid-level predictions.
        :type mid_predictions: Dict[str, float]
        :return: Combined predictions.
        :rtype: Dict[str, float]
        """
        combined = {}
        for metric in set(low_predictions.keys()) | set(mid_predictions.keys()):
            low = low_predictions.get(metric, 0.0)
            mid = mid_predictions.get(metric, 0.0)
            combined[metric] = (low + mid) / 2.0
        return combined

    def predict(self, low_predictions: Dict[str, float],
                mid_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Return combined predictions and recommendations.

        :param low_predictions: Low-level predictions.
        :type low_predictions: Dict[str, float]
        :param mid_predictions: Mid-level predictions.
        :type mid_predictions: Dict[str, float]
        :return: Prediction result dictionary.
        :rtype: Dict[str, Any]
        """
        combined = self._combine(low_predictions, mid_predictions)
        anomaly_likelihood = self._compute_anomaly_likelihood(combined)
        recommended_action = self._recommend_action(combined, anomaly_likelihood)
        return self._build_prediction_result(
            combined, anomaly_likelihood, recommended_action
        )

    def _build_prediction_result(
        self, combined: Dict[str, float],
        anomaly_likelihood: float, recommended_action: str
    ) -> Dict[str, Any]:
        """Build prediction result dict.

        :param combined: Combined predictions.
        :type combined: Dict[str, float]
        :param anomaly_likelihood: Anomaly likelihood.
        :type anomaly_likelihood: float
        :param recommended_action: Recommended action.
        :type recommended_action: str
        :return: Prediction result dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'combined_predictions': combined,
            'anomaly_likelihood': anomaly_likelihood,
            'recommended_action': recommended_action,
            'confidence': 1.0 - anomaly_likelihood,
        }

    def update(self, state: Dict[str, Any], action_taken: str,
               outcome: float) -> None:
        """Update strategy scores.

        :param state: Current state.
        :type state: Dict[str, Any]
        :param action_taken: Action taken.
        :type action_taken: str
        :param outcome: Outcome value.
        :type outcome: float
        :return: None
        :rtype: None
        """
        self._append_abstract_state(state, action_taken, outcome)
        self._update_strategy_score(action_taken, outcome)

    def _append_abstract_state(self, state: Dict[str, Any],
                               action_taken: str, outcome: float) -> None:
        """Append current state to abstract states history.

        :param state: Current state.
        :type state: Dict[str, Any]
        :param action_taken: Action taken.
        :type action_taken: str
        :param outcome: Outcome value.
        :type outcome: float
        :return: None
        :rtype: None
        """
        self._abstract_states.append({
            'timestamp': time.time(),
            'state_summary': self._summarize_state(state),
            'action': action_taken,
            'outcome': outcome,
        })

    def _update_strategy_score(self, action_taken: str, outcome: float) -> None:
        """Update score for a strategy.

        :param action_taken: Action taken.
        :type action_taken: str
        :param outcome: Outcome value.
        :type outcome: float
        :return: None
        :rtype: None
        """
        if action_taken not in self._strategy_scores:
            self._strategy_scores[action_taken] = 0.5
        self._strategy_scores[action_taken] = (
            0.9 * self._strategy_scores[action_taken] + 0.1 * outcome
        )

    def _summarize_state(self, state: Dict[str, Any]) -> str:
        """Summarize system state.

        :param state: Current state.
        :type state: Dict[str, Any]
        :return: State summary string.
        :rtype: str
        """
        cpu, mem = self._extract_cpu_mem(state)
        return self._classify_load(cpu, mem)

    def _extract_cpu_mem(self, state: Dict[str, Any]) -> tuple:
        """Extract cpu and mem percentages from state.

        :param state: Current state.
        :type state: Dict[str, Any]
        :return: CPU and memory percentages.
        :rtype: tuple
        """
        system = state.get('system', {})
        cpu = system.get('cpu_percent', 0.0)
        mem = system.get('mem_percent', 0.0)
        return cpu, mem

    def _classify_load(self, cpu: float, mem: float) -> str:
        """Classify system load based on cpu and mem.

        :param cpu: CPU percentage.
        :type cpu: float
        :param mem: Memory percentage.
        :type mem: float
        :return: Load classification string.
        :rtype: str
        """
        if cpu > 80.0 or mem > 80.0:
            return 'high_load'
        elif cpu < 20.0 and mem < 30.0:
            return 'low_load'
        else:
            return 'normal'

    def _compute_anomaly_likelihood(self, predictions: Dict[str, float]) -> float:
        """Compute anomaly likelihood.

        :param predictions: Predictions dictionary.
        :type predictions: Dict[str, float]
        :return: Anomaly likelihood.
        :rtype: float
        """
        values = list(predictions.values())
        if not values:
            return 0.0
        max_val = max(values)
        return min(1.0, max_val * 2.0)

    def _recommend_action(self, predictions: Dict[str, float],
                          anomaly: float) -> str:
        """Recommend action based on anomaly.

        :param predictions: Predictions dictionary.
        :type predictions: Dict[str, float]
        :param anomaly: Anomaly likelihood.
        :type anomaly: float
        :return: Recommended action string.
        :rtype: str
        """
        if anomaly > 0.7:
            return 'emergency_safeguard'
        elif anomaly > 0.4:
            return 'increase_monitoring'
        else:
            return 'continue_exploration'

    def get_best_strategy(self) -> Optional[str]:
        """Return best strategy.

        :param: 无
        :type: 无
        :return: Best strategy name or None.
        :rtype: Optional[str]
        """
        if not self._strategy_scores:
            return None
        return max(self._strategy_scores, key=self._strategy_scores.get)
