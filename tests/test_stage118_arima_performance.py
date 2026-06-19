"""阶段118：ARIMA预测器性能瓶颈攻坚测试。

验证内容：
1. fit_interval缓存机制生效（减少重复拟合）
2. fit_timeout超时保护生效（避免阻塞）
3. 超时后降级为均值预测
4. 预测结果仍然合理
5. 性能提升可量化
6. set_params正确重置缓存
7. maxlen=100限制生效
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.arima_predictor import ARIMAModel


class TestARIMAFitInterval(unittest.TestCase):
    """测试fit_interval缓存机制。"""

    def test_cache_reduces_refit_count(self):
        """fit_interval=5时，50次predict应只refit约10次。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=5, fit_timeout=10.0)
        for i in range(10):
            model.update(float(i) * 0.5)
        refit_count = 0
        for i in range(50):
            model.update(float(10 + i) * 0.5)
            before_hash = model._cached_hash
            model.predict(steps=1)
            after_hash = model._cached_hash
            # hash变化说明发生了refit
            if before_hash != after_hash:
                refit_count += 1
        # 50次predict，fit_interval=5，应约refit 10次（允许±2误差）
        self.assertLess(refit_count, 15, f"refit_count={refit_count}过高，缓存未生效")
        self.assertGreater(refit_count, 5, f"refit_count={refit_count}过低，可能未正常refit")

    def test_fit_interval_1_means_always_refit(self):
        """fit_interval=1时，每次predict都应refit。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=1, fit_timeout=10.0)
        for i in range(10):
            model.update(float(i) * 0.5)
        refit_count = 0
        for i in range(10):
            model.update(float(10 + i) * 0.5)
            before_hash = model._cached_hash
            model.predict(steps=1)
            after_hash = model._cached_hash
            if before_hash != after_hash:
                refit_count += 1
        self.assertEqual(refit_count, 10, f"fit_interval=1应每次refit，实际{refit_count}")

    def test_performance_improvement(self):
        """50次predict耗时应显著低于无缓存时。"""
        # 有缓存
        model_cached = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=5, fit_timeout=10.0)
        for i in range(10):
            model_cached.update(float(i) * 0.5)
        start = time.time()
        for i in range(50):
            model_cached.update(float(10 + i) * 0.5)
            model_cached.predict(steps=1)
        cached_time = time.time() - start

        # 无缓存（fit_interval=1）
        model_nocache = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=1, fit_timeout=10.0)
        for i in range(10):
            model_nocache.update(float(i) * 0.5)
        start = time.time()
        for i in range(50):
            model_nocache.update(float(10 + i) * 0.5)
            model_nocache.predict(steps=1)
        nocache_time = time.time() - start

        # 缓存版本应至少快2倍
        self.assertLess(cached_time * 2, nocache_time,
                        f"缓存版本{cached_time:.3f}s未显著快于无缓存{nocache_time:.3f}s")


class TestARIMAFitTimeout(unittest.TestCase):
    """测试fit_timeout超时保护。"""

    def test_timeout_falls_back_to_avg(self):
        """fit超时应降级为均值预测。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=1, fit_timeout=0.001)
        for i in range(20):
            model.update(float(i) * 0.5)
        # 用mock让fit变慢，确保超时
        import time as _time
        original_fit = None

        def slow_fit(self_inner):
            _time.sleep(0.5)
            return original_fit(self_inner)

        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        original_fit = _ARIMA.fit
        try:
            _ARIMA.fit = slow_fit
            result = model.predict(steps=1)
            self.assertEqual(len(result), 1)
            # 超时降级为均值
            avg = sum(model._history) / len(model._history)
            self.assertAlmostEqual(result[0], avg, places=4)
        finally:
            _ARIMA.fit = original_fit

    def test_normal_fit_not_affected_by_timeout(self):
        """正常fit不应被合理timeout影响。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=1, fit_timeout=30.0)
        for i in range(20):
            model.update(float(i) * 0.5)
        result = model.predict(steps=1)
        self.assertEqual(len(result), 1)
        # 应有缓存模型（说明fit成功未超时）
        self.assertIsNotNone(model._cached_fitted)


class TestARIMAMaxlenLimit(unittest.TestCase):
    """测试maxlen限制。"""

    def test_history_capped_at_100(self):
        """历史数据应被限制在100条。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        for i in range(200):
            model.update(float(i))
        self.assertEqual(len(model._history), 100)

    def test_old_data_dropped(self):
        """超过100条后，旧数据应被丢弃。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        for i in range(150):
            model.update(float(i))
        # 最旧的应为50（0-49已被丢弃）
        self.assertEqual(list(model._history)[0], 50.0)


class TestARIMACacheReset(unittest.TestCase):
    """测试set_params正确重置缓存。"""

    def test_set_params_resets_cache(self):
        """set_params应清空缓存。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=5)
        for i in range(20):
            model.update(float(i) * 0.5)
        model.predict(steps=1)
        self.assertIsNotNone(model._cached_fitted)
        self.assertEqual(model._updates_since_fit, 0)

        # set_params应重置缓存
        model.set_params({'history': [1.0, 2.0, 3.0, 4.0, 5.0]})
        self.assertIsNone(model._cached_fitted)
        self.assertEqual(model._updates_since_fit, 0)

    def test_update_increments_counter(self):
        """update应递增_updates_since_fit。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=5)
        self.assertEqual(model._updates_since_fit, 0)
        model.update(1.0)
        self.assertEqual(model._updates_since_fit, 1)
        model.update(2.0)
        self.assertEqual(model._updates_since_fit, 2)


class TestARIMAFallbackCorrectness(unittest.TestCase):
    """测试降级逻辑正确性。"""

    def test_fallback_when_statsmodels_missing(self):
        """statsmodels不可用时应降级为均值。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        for i in range(20):
            model.update(float(i) * 0.5)
        # 模拟statsmodels不可用：同时阻止类缓存和重新导入
        original_class = ARIMAModel._arima_class
        original_import = ARIMAModel._try_import_arima
        try:
            ARIMAModel._arima_class = None
            ARIMAModel._try_import_arima = lambda self: None
            result = model.predict(steps=1)
            self.assertEqual(len(result), 1)
            avg = sum(model._history) / len(model._history)
            self.assertAlmostEqual(result[0], avg, places=4)
        finally:
            ARIMAModel._arima_class = original_class
            ARIMAModel._try_import_arima = original_import

    def test_min_samples_uses_fallback(self):
        """数据不足min_samples时应使用均值降级。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=10)
        for i in range(5):
            model.update(float(i))
        result = model.predict(steps=1)
        avg = sum(model._history) / len(model._history)
        self.assertAlmostEqual(result[0], avg, places=4)
        # 不应有缓存模型
        self.assertIsNone(model._cached_fitted)

    def test_nan_value_ignored(self):
        """NaN值应被忽略。"""
        import math
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        model.update(1.0)
        model.update(float('nan'))
        model.update(2.0)
        self.assertEqual(len(model._history), 2)

    def test_inf_value_ignored(self):
        """Inf值应被忽略。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        model.update(1.0)
        model.update(float('inf'))
        model.update(2.0)
        self.assertEqual(len(model._history), 2)


class TestARIMAIntegrationPerformance(unittest.TestCase):
    """集成性能测试：模拟真实agent循环。"""

    def test_100_cycle_performance(self):
        """100次update+predict循环应在30秒内完成。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=5, fit_timeout=5.0)
        for i in range(10):
            model.update(float(i) * 0.5)
        start = time.time()
        for i in range(100):
            model.update(float(10 + i) * 0.5)
            model.predict(steps=1)
        elapsed = time.time() - start
        self.assertLess(elapsed, 30.0, f"100 cycles took {elapsed:.2f}s, expected <30s")

    def test_prediction_reasonable_after_many_updates(self):
        """多次update后预测值应仍然合理。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5, fit_interval=5, fit_timeout=5.0)
        for i in range(50):
            model.update(float(i) * 1.0)
        result = model.predict(steps=1)
        self.assertEqual(len(result), 1)
        # 预测值应在数据范围内
        self.assertGreater(result[0], 0)
        self.assertLess(result[0], 100)


if __name__ == '__main__':
    unittest.main()
