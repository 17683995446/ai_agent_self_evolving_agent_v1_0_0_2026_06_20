import abc
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseEvaluator(abc.ABC):
    """Abstract base class for evaluators."""

    @abc.abstractmethod
    def evaluate(self, predictions: Dict[str, float], actuals: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate predictions against actuals.

        :param predictions: Predicted values dictionary.
        :type predictions: Dict[str, float]
        :param actuals: Actual values dictionary.
        :type actuals: Dict[str, float]
        :return: Evaluation result dictionary.
        :rtype: Dict[str, Any]
        """

    @abc.abstractmethod
    def evaluate_single(self, prediction: float, actual: float) -> Dict[str, float]:
        """Evaluate a single prediction.

        :param prediction: Predicted value.
        :type prediction: float
        :param actual: Actual value.
        :type actual: float
        :return: Evaluation result dictionary.
        :rtype: Dict[str, float]
        """


class ErrorEvaluator(BaseEvaluator):
    """Evaluator for prediction errors."""

    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        """Initialize the error evaluator.

        :param metrics: List of metric names.
        :type metrics: Optional[List[str]]
        :return: None
        :rtype: None
        """
        self.metrics = metrics or []

    def _compute_errors(self, predictions: Dict[str, float],
                        actuals: Dict[str, float]) -> Dict[str, float]:
        """Compute absolute errors per metric.

        :param predictions: Predicted values dictionary.
        :type predictions: Dict[str, float]
        :param actuals: Actual values dictionary.
        :type actuals: Dict[str, float]
        :return: Errors dictionary.
        :rtype: Dict[str, float]
        """
        errors = {}
        for metric, pred in predictions.items():
            actual = actuals.get(metric, 0.0)
            errors[metric] = abs(actual - pred)
        return errors

    def _compute_mae(self, errors: Dict[str, float]) -> float:
        """Compute mean absolute error.

        :param errors: Errors dictionary.
        :type errors: Dict[str, float]
        :return: Mean absolute error.
        :rtype: float
        """
        return float(np.mean(list(errors.values()))) if errors else 0.0

    def _compute_mse(self, errors: Dict[str, float]) -> float:
        """Compute mean squared error.

        :param errors: Errors dictionary.
        :type errors: Dict[str, float]
        :return: Mean squared error.
        :rtype: float
        """
        return float(np.mean([e ** 2 for e in errors.values()])) if errors else 0.0

    def _compute_max_error(self, errors: Dict[str, float]) -> float:
        """Compute maximum error.

        :param errors: Errors dictionary.
        :type errors: Dict[str, float]
        :return: Maximum error.
        :rtype: float
        """
        return float(max(errors.values())) if errors else 0.0

    def evaluate(self, predictions: Dict[str, float], actuals: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate predictions against actuals.

        :param predictions: Predicted values dictionary.
        :type predictions: Dict[str, float]
        :param actuals: Actual values dictionary.
        :type actuals: Dict[str, float]
        :return: Evaluation result dictionary.
        :rtype: Dict[str, Any]
        """
        logger.debug("ErrorEvaluator.evaluate started")
        predictions = predictions or {}
        actuals = actuals or {}
        errors = self._compute_errors(predictions, actuals)
        result = self._build_result(errors)
        logger.debug("ErrorEvaluator.evaluate done, mae=%.4f", result['mae'])
        return result

    def _build_result(self, errors: Dict[str, float]) -> Dict[str, Any]:
        """Build evaluation result dict from errors.

        :param errors: Errors dictionary.
        :type errors: Dict[str, float]
        :return: Evaluation result dictionary.
        :rtype: Dict[str, Any]
        """
        mae = self._compute_mae(errors)
        mse = self._compute_mse(errors)
        max_error = self._compute_max_error(errors)
        return {
            'errors': errors,
            'mae': round(mae, 6),
            'mse': round(mse, 6),
            'max_error': round(max_error, 6),
            'metric_count': len(errors),
        }

    def evaluate_single(self, prediction: float, actual: float) -> Dict[str, float]:
        """Evaluate a single prediction.

        :param prediction: Predicted value.
        :type prediction: float
        :param actual: Actual value.
        :type actual: float
        :return: Evaluation result dictionary.
        :rtype: Dict[str, float]
        """
        error = abs(actual - prediction)
        return {
            'error': round(error, 6),
            'squared_error': round(error ** 2, 6),
            'relative_error': round(error / max(abs(actual), 1e-10), 6),
        }
