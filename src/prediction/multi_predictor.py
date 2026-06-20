import abc
from typing import Any, Dict, List, Optional

from .arima_predictor import ARIMAModel
from .base_predictor import AdaptiveExponentialSmoother, ExponentialSmoother, SimpleMovingAverage


class BaseMultiPredictor(abc.ABC):
    """Abstract base class for multi-metric predictors."""

    @abc.abstractmethod
    def predict_all(self) -> Dict[str, float]:
        """Return predictions for all metrics.

        :param: 无
        :type: 无
        :return: Predictions dictionary.
        :rtype: Dict[str, float]
        """

    @abc.abstractmethod
    def update_all(self, actual_values: Dict[str, float]) -> None:
        """Update all predictors with actual values.

        :param actual_values: Actual values dictionary.
        :type actual_values: Dict[str, float]
        :return: None
        :rtype: None
        """


class MultiPredictor(BaseMultiPredictor):
    """Multi-metric predictor supporting multiple model types."""

    def __init__(self, metrics: Optional[List[str]] = None,
                 model_type: str = 'adaptive_exponential',
                 model_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the multi-predictor.

        :param metrics: List of metric names.
        :type metrics: Optional[List[str]]
        :param model_type: Model type name.
        :type model_type: str
        :param model_config: Model configuration dictionary.
        :type model_config: Optional[Dict[str, Any]]
        :return: None
        :rtype: None
        """
        self.metrics = metrics or self._default_metrics()
        self.model_type = model_type
        self.model_config = model_config or {}
        self.predictors: Dict[str, Any] = {}
        self._init_predictors()

    @staticmethod
    def _default_metrics() -> List[str]:
        """Return default metric names.

        :param: 无
        :type: 无
        :return: Default metric names.
        :rtype: List[str]
        """
        return [
            'system.cpu_percent',
            'system.mem_percent',
            'system.disk_percent',
            'system.net_sent_mb_s',
            'system.net_recv_mb_s',
            'logs.error_count_1m',
            'logs.warn_count_1m',
            'files.recent_changes_1m',
        ]

    def _init_predictors(self) -> None:
        """Initialize predictors for each metric.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        for metric in self.metrics:
            self.predictors[metric] = self._create_model()

    def _create_exponential(self) -> ExponentialSmoother:
        """Create exponential smoother.

        :param: 无
        :type: 无
        :return: Exponential smoother instance.
        :rtype: ExponentialSmoother
        """
        return ExponentialSmoother(
            alpha=self.model_config.get('alpha', 0.3)
        )

    def _create_adaptive(self) -> AdaptiveExponentialSmoother:
        """Create adaptive exponential smoother.

        :param: 无
        :type: 无
        :return: Adaptive exponential smoother instance.
        :rtype: AdaptiveExponentialSmoother
        """
        return AdaptiveExponentialSmoother(
            alpha_init=self.model_config.get('alpha', 0.3),
            alpha_min=self.model_config.get('alpha_min', 0.05),
            alpha_max=self.model_config.get('alpha_max', 0.8),
            adaptation_rate=self.model_config.get('adaptation_rate', 0.01),
        )

    def _create_sma(self) -> SimpleMovingAverage:
        """Create simple moving average.

        :param: 无
        :type: 无
        :return: Simple moving average instance.
        :rtype: SimpleMovingAverage
        """
        return SimpleMovingAverage(
            window=self.model_config.get('window', 10)
        )

    def _create_arima(self) -> ARIMAModel:
        """Create ARIMA model.

        :param: 无
        :type: 无
        :return: ARIMA model instance.
        :rtype: ARIMAModel
        """
        return ARIMAModel(
            order=tuple(self.model_config.get('order', [1, 1, 1])),
            min_samples=self.model_config.get('min_samples', 10)
        )

    def _create_model(self) -> Any:
        """Create model based on model_type.

        :param: 无
        :type: 无
        :return: Model instance.
        :rtype: Any
        """
        creator = self._model_creators().get(self.model_type)
        if creator is not None:
            return creator()
        return ExponentialSmoother(alpha=0.3)

    def _model_creators(self) -> Dict[str, Any]:
        """Return mapping of model types to creator methods.

        :param: 无
        :type: 无
        :return: Mapping dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'exponential': self._create_exponential,
            'adaptive_exponential': self._create_adaptive,
            'sma': self._create_sma,
            'arima': self._create_arima,
        }

    def predict_all(self) -> Dict[str, float]:
        """Return predictions for all metrics.

        :param: 无
        :type: 无
        :return: Predictions dictionary.
        :rtype: Dict[str, float]
        """
        predictions: Dict[str, float] = {}
        for metric, predictor in self.predictors.items():
            preds = predictor.predict(steps=1)
            predictions[metric] = float(preds[0]) if preds else 0.0
        return predictions

    def update_all(self, actual_values: Dict[str, float]) -> None:
        """Update all predictors with actual values.

        :param actual_values: Actual values dictionary.
        :type actual_values: Dict[str, float]
        :return: None
        :rtype: None
        """
        for metric, value in actual_values.items():
            if metric in self.predictors:
                self.predictors[metric].update(float(value))

    def add_metric(self, metric: str) -> None:
        """Add a new metric.

        :param metric: Metric name.
        :type metric: str
        :return: None
        :rtype: None
        """
        if metric not in self.predictors:
            self.predictors[metric] = self._create_model()
            self.metrics.append(metric)

    def remove_metric(self, metric: str) -> None:
        """Remove a metric.

        :param metric: Metric name.
        :type metric: str
        :return: None
        :rtype: None
        """
        if metric in self.predictors:
            del self.predictors[metric]
            if metric in self.metrics:
                self.metrics.remove(metric)

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
