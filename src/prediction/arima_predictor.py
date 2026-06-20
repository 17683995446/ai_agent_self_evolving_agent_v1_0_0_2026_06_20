from typing import List, Optional, Dict, Any
from collections import deque
import logging
import threading

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class ARIMAModel(BasePredictor):
    """ARIMA time-series predictor."""

    _arima_class = None

    def __init__(self, order: tuple = (1, 1, 1), min_samples: int = 10,
                 fit_interval: int = 5, fit_timeout: float = 5.0) -> None:
        """Initialize the ARIMA model.

        :param order: ARIMA model order (p, d, q).
        :type order: tuple
        :param min_samples: Minimum samples required before fitting.
        :type min_samples: int
        :param fit_interval: Refit every N updates; reuse cached model in between.
        :type fit_interval: int
        :param fit_timeout: Max seconds for a single fit; fall back on timeout.
        :type fit_timeout: float
        :return: None
        :rtype: None
        """
        self.order = order
        self.min_samples = min_samples
        self.fit_interval = max(1, fit_interval)
        self.fit_timeout = max(0.1, fit_timeout)
        self._init_history()
        self._init_cache()

    def _init_history(self) -> None:
        """Initialize history deque.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._history: deque = deque(maxlen=100)

    def _init_cache(self) -> None:
        """Initialize fitted model cache.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._fitted = False
        self._cached_fitted = None
        self._cached_hash = None
        self._updates_since_fit = 0

    def _get_arima_class(self) -> Optional[Any]:
        """Lazy-load and cache ARIMA class.

        :param: 无
        :type: 无
        :return: ARIMA class or None if not available.
        :rtype: Optional[Any]
        """
        if ARIMAModel._arima_class is not None:
            return ARIMAModel._arima_class
        return self._try_import_arima()

    def _try_import_arima(self) -> Optional[Any]:
        """Attempt to import statsmodels ARIMA.

        :param: 无
        :type: 无
        :return: ARIMA class or None if import fails.
        :rtype: Optional[Any]
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            ARIMAModel._arima_class = ARIMA
            return ARIMA
        except ImportError:
            return None

    def _fallback_avg(self, steps: int) -> List[float]:
        """Return average-based fallback prediction.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Fallback predicted values.
        :rtype: List[float]
        """
        if len(self._history) == 0:
            return [0.0] * steps
        avg = sum(self._history) / len(self._history)
        return [avg] * steps

    def _try_cached_forecast(self, steps: int) -> Optional[List[float]]:
        """Try to reuse cached fitted model.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Forecasted values or None if cache miss.
        :rtype: Optional[List[float]]
        """
        if self._cached_fitted is None:
            return None
        return self._forecast_from_cached(steps)

    def _forecast_from_cached(self, steps: int) -> Optional[List[float]]:
        """Forecast using cached fitted model.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Forecasted values or None on error.
        :rtype: Optional[List[float]]
        """
        try:
            forecast = self._cached_fitted.forecast(steps=steps)
            return [float(v) for v in forecast]
        except Exception as e:
            logger.debug("ARIMAModel._forecast_from_cached: cache forecast failed: %s", e)
            return None

    def _fit_and_forecast(self, steps: int) -> List[float]:
        """Fit ARIMA and return forecast.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Forecasted values.
        :rtype: List[float]
        """
        ARIMA = self._get_arima_class()
        if ARIMA is None:
            return self._fallback_avg(steps)
        return self._fit_model_and_forecast(ARIMA, steps)

    def _fit_model_and_forecast(self, ARIMA: Any, steps: int) -> List[float]:
        """Build model, fit, cache, and forecast.

        :param ARIMA: ARIMA class.
        :type ARIMA: Any
        :param steps: Number of steps to predict.
        :type steps: int
        :return: Forecasted values.
        :rtype: List[float]
        """
        series = list(self._history)
        try:
            return self._build_fit_forecast(ARIMA, series, steps)
        except Exception as e:
            logger.debug("ARIMAModel._fit_model_and_forecast: fit failed: %s", e)
            return self._fallback_avg(steps)

    def _build_fit_forecast(self, ARIMA: Any, series: List[float], steps: int) -> List[float]:
        """Build ARIMA model, fit, cache, and forecast.

        :param ARIMA: ARIMA class.
        :type ARIMA: Any
        :param series: Time series data.
        :type series: List[float]
        :param steps: Number of steps to predict.
        :type steps: int
        :return: Forecasted values.
        :rtype: List[float]
        """
        model = ARIMA(series, order=self.order)
        fitted = self._fit_with_timeout(model)
        if fitted is None:
            logger.warning("ARIMAModel._build_fit_forecast: fit timed out, using fallback")
            return self._fallback_avg(steps)
        self._cache_fitted(fitted, series)
        forecast = fitted.forecast(steps=steps)
        return [float(v) for v in forecast]

    def _fit_with_timeout(self, model: Any) -> Optional[Any]:
        """Fit ARIMA model with timeout protection.

        :param model: Unfitted ARIMA model.
        :type model: Any
        :return: Fitted model or None on timeout/failure.
        :rtype: Optional[Any]
        """
        result: Dict[str, Any] = {'fitted': None, 'error': None}
        def _do_fit():
            try:
                result['fitted'] = model.fit()
            except Exception as e:
                result['error'] = e
        t = threading.Thread(target=_do_fit, daemon=True)
        t.start()
        t.join(timeout=self.fit_timeout)
        if t.is_alive():
            logger.warning("ARIMAModel._fit_with_timeout: fit exceeded %.1fs", self.fit_timeout)
            return None
        if result['error'] is not None:
            raise result['error']
        return result['fitted']

    def _cache_fitted(self, fitted: Any, series: List[float]) -> None:
        """Cache fitted model and reset update counter.

        :param fitted: Fitted ARIMA model.
        :type fitted: Any
        :param series: Time series data.
        :type series: List[float]
        :return: None
        :rtype: None
        """
        self._cached_fitted = fitted
        self._cached_hash = hash(tuple(series))
        self._updates_since_fit = 0

    def predict(self, steps: int = 1) -> List[float]:
        """Predict future values.

        :param steps: Number of steps to predict.
        :type steps: int
        :return: Predicted future values.
        :rtype: List[float]
        """
        if len(self._history) < self.min_samples:
            return self._fallback_avg(steps)
        # 修复阶段118: 仅在达到fit_interval时重新拟合，否则复用缓存模型
        if self._cached_fitted is not None and self._updates_since_fit < self.fit_interval:
            cached = self._forecast_from_cached(steps)
            if cached is not None:
                return cached
        return self._fit_and_forecast(steps)

    def update(self, actual: float) -> None:
        """Update predictor with actual value.

        :param actual: Actual observed value.
        :type actual: float
        :return: None
        :rtype: None
        """
        import math
        if actual is None or math.isnan(actual) or math.isinf(actual):
            return
        self._history.append(actual)
        self._updates_since_fit += 1

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters.

        :param: 无
        :type: 无
        :return: Current parameters dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'order': self.order,
            'min_samples': self.min_samples,
            'history': list(self._history),
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters from dict.

        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self._apply_order(params)
        self._apply_min_samples(params)
        self._apply_history(params)

    def _apply_order(self, params: Dict[str, Any]) -> None:
        """Apply order parameter.

        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.order = params.get('order', self.order)

    def _apply_min_samples(self, params: Dict[str, Any]) -> None:
        """Apply min_samples parameter.

        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.min_samples = params.get('min_samples', self.min_samples)

    def _apply_history(self, params: Dict[str, Any]) -> None:
        """Apply history parameter and clear cache.

        :param params: Parameters dictionary.
        :type params: Dict[str, Any]
        :return: None
        :rtype: None
        """
        if 'history' in params:
            self._history.clear()
            self._history.extend(params['history'])
            self._cached_fitted = None
            self._cached_hash = None
            self._updates_since_fit = 0
