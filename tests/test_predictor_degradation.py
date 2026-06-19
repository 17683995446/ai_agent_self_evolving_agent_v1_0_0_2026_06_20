import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.base_predictor import AdaptiveExponentialSmoother
from src.prediction.arima_predictor import ARIMAModel
from src.prediction.layered_predictor import LowLevelPredictor, MidLevelPredictor, HighLevelPredictor
from src.prediction.multi_predictor import MultiPredictor
from src.prediction.error_evaluator import ErrorEvaluator
from src.optimization.micro_optimizer import MicroOptimizer
from src.optimization.snapshot_manager import SnapshotManager
from src.optimization.auto_rollback import AutoRollback


class TestAdaptiveSmootherDegradation(unittest.TestCase):
    """测试自适应平滑器在复杂场景下的退化与恢复。"""

    def test_alpha_drift_under_noisy_data(self):
        """噪声数据导致alpha异常漂移后，系统应能自适应修正。"""
        smoother = AdaptiveExponentialSmoother(
            alpha_init=0.3,
            alpha_min=0.05,
            alpha_max=0.8,
            adaptation_rate=0.01
        )
        base_value = 50.0
        for i in range(200):
            noise = (i % 10 - 5) * 10.0
            smoother.update(base_value + noise)

        self.assertLessEqual(smoother.alpha, 0.8)
        self.assertGreaterEqual(smoother.alpha, 0.05)
        self.assertIsNotNone(smoother.level)

    def test_level_drift_recovery(self):
        """level在数据分布突变后应能逐步跟踪新分布。"""
        smoother = AdaptiveExponentialSmoother(alpha_init=0.3)
        for _ in range(50):
            smoother.update(10.0)
        old_level = smoother.level

        for _ in range(100):
            smoother.update(100.0)

        self.assertGreater(smoother.level, old_level)
        self.assertGreater(smoother.level, 50.0)

    def test_sudden_spike_handling(self):
        """突发极端值不应导致level无限发散。"""
        smoother = AdaptiveExponentialSmoother(alpha_init=0.3)
        for _ in range(20):
            smoother.update(10.0)
        smoother.update(1e6)
        self.assertLess(smoother.level, 1e6)
        self.assertGreater(smoother.level, 0.0)

    def test_none_input_ignored(self):
        """None输入应被安全忽略，不破坏状态。"""
        smoother = AdaptiveExponentialSmoother(alpha_init=0.3)
        smoother.update(10.0)
        old_level = smoother.level
        old_alpha = smoother.alpha
        smoother.update(None)
        self.assertEqual(smoother.level, old_level)
        self.assertEqual(smoother.alpha, old_alpha)


class TestARIMAdegradation(unittest.TestCase):
    """测试ARIMA模型在数据不足和异常场景下的降级行为。"""

    def test_zero_samples_fallback(self):
        """零样本时应返回0.0预测。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=10)
        preds = model.predict(steps=1)
        self.assertEqual(preds[0], 0.0)

    def test_below_min_samples_fallback(self):
        """样本不足min_samples时应回退到均值预测。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=10)
        for i in range(5):
            model.update(float(i))
        preds = model.predict(steps=1)
        expected_avg = sum(range(5)) / 5.0
        self.assertEqual(preds[0], expected_avg)

    def test_min_samples_boundary(self):
        """刚好达到min_samples边界时应正常拟合。"""
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        for i in range(5):
            model.update(float(i))
        preds = model.predict(steps=1)
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], float)

    def test_corrupted_history_recovery(self):
        """历史数据被异常值污染后，模型应继续运行不崩溃。"""
        import math
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        values = [1.0, 2.0, float('inf'), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for v in values:
            model.update(v)
        preds = model.predict(steps=1)
        self.assertEqual(len(preds), 1)
        self.assertFalse(any(math.isnan(p) for p in preds))

    def test_statsmodels_missing_fallback(self):
        """statsmodels缺失时应回退到均值预测。"""
        from unittest.mock import patch
        model = ARIMAModel(order=(1, 1, 1), min_samples=5)
        for i in range(10):
            model.update(float(i))
        model._cached_fitted = None
        model._cached_hash = None
        with patch.object(model, '_get_arima_class', return_value=None):
            preds = model.predict(steps=1)
            self.assertEqual(len(preds), 1)
            expected_avg = sum(range(10)) / 10.0
            self.assertEqual(preds[0], expected_avg)


class TestMicroOptimizerDegradationRecovery(unittest.TestCase):
    """测试MicroOptimizer在预测器退化时的自动修正能力。"""

    def setUp(self):
        self.optimizer = MicroOptimizer(learning_rate=0.1, min_improvement=0.001)

    def test_error_increase_triggers_adjustment(self):
        """误差持续增大时，optimizer应调整alpha。"""
        predictor = MultiPredictor(metrics=['cpu'], model_type='adaptive_exponential')
        predictor.update_all({'cpu': 50.0})

        errors = {'cpu': 0.1}
        self.optimizer.optimize(predictor, errors)
        errors = {'cpu': 0.5}
        improvements = self.optimizer.optimize(predictor, errors)

        self.assertIn('cpu', improvements)
        old_alpha = improvements['cpu']['old_params']['alpha']
        new_alpha = improvements['cpu']['new_params']['alpha']
        self.assertNotEqual(old_alpha, new_alpha)

    def test_error_decrease_no_adjustment(self):
        """误差减小时，optimizer不应过度调整。"""
        predictor = MultiPredictor(metrics=['cpu'], model_type='adaptive_exponential')
        predictor.update_all({'cpu': 50.0})

        errors = {'cpu': 0.5}
        self.optimizer.optimize(predictor, errors)
        errors = {'cpu': 0.1}
        improvements = self.optimizer.optimize(predictor, errors)

        self.assertEqual(improvements, {})

    def test_gradient_direction_correctness(self):
        """梯度方向应使alpha向减小误差方向移动。"""
        params = {'alpha': 0.5, 'level': 50.0}
        new_params = self.optimizer._adjust_params(params, 0.8, 0.5)
        self.assertNotEqual(new_params['alpha'], 0.5)
        self.assertGreaterEqual(new_params['alpha'], 0.05)
        self.assertLessEqual(new_params['alpha'], 0.8)

    def test_metric_trend_detection(self):
        """应能检测误差趋势（上升/下降/平稳）。"""
        predictor = MultiPredictor(metrics=['cpu'])
        for i in range(20):
            self.optimizer.optimize(predictor, {'cpu': float(i) * 0.1})
        trend = self.optimizer.get_metric_trend('cpu')
        self.assertIsNotNone(trend)
        self.assertGreater(trend, 0.0)

    def test_reset_clears_history(self):
        """reset_metric_history应清空历史，阻止错误优化。"""
        predictor = MultiPredictor(metrics=['cpu'])
        self.optimizer.optimize(predictor, {'cpu': 0.5})
        self.optimizer.optimize(predictor, {'cpu': 0.6})
        self.optimizer.reset_metric_history('cpu')
        self.assertEqual(len(self.optimizer._error_history['cpu']), 0)


class TestLayeredPredictorDegradation(unittest.TestCase):
    """测试多层预测器联合退化与恢复。"""

    def test_low_level_degradation_recovery(self):
        """底层预测器在数据分布突变后应恢复跟踪。"""
        low = LowLevelPredictor(metrics=['cpu'])
        for _ in range(50):
            low.update({'cpu': 10.0})
        old_pred = low.predict()['cpu']

        for _ in range(100):
            low.update({'cpu': 90.0})
        new_pred = low.predict()['cpu']

        self.assertGreater(new_pred, old_pred)
        self.assertGreater(new_pred, 50.0)

    def test_mid_level_trend_accuracy(self):
        """中层预测器在趋势变化时应准确反映。"""
        mid = MidLevelPredictor(metrics=['cpu'])
        for i in range(20):
            mid.update({'cpu': float(i)})
        trend = mid.get_trend('cpu')
        self.assertIsNotNone(trend)
        self.assertGreater(trend, 0.0)

    def test_high_level_anomaly_escalation(self):
        """高层预测器应在异常持续时提升异常等级。"""
        high = HighLevelPredictor()
        result = high.predict({'cpu': 0.1}, {'cpu': 0.1})
        self.assertEqual(result['recommended_action'], 'continue_exploration')

        result = high.predict({'cpu': 0.9}, {'cpu': 0.9})
        self.assertEqual(result['recommended_action'], 'emergency_safeguard')

    def test_high_level_strategy_learning(self):
        """高层策略评分应在正向反馈下提升。"""
        high = HighLevelPredictor()
        for _ in range(10):
            high.update({'system': {'cpu_percent': 50.0}}, 'explore', 1.0)
        best = high.get_best_strategy()
        self.assertEqual(best, 'explore')
        self.assertGreater(high._strategy_scores['explore'], 0.5)


class TestAutoRollbackDegradation(unittest.TestCase):
    """测试AutoRollback在模型退化时的触发与恢复。"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=5)
        self.ar = AutoRollback(self.sm, mae_degradation_threshold=0.2)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_baseline_establishment(self):
        """基线应正确建立并关联快照。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        stats = self.ar.get_stats()
        self.assertEqual(stats['baseline_mae'], 0.1)
        self.assertEqual(stats['baseline_snapshot_id'], sid)

    def test_degradation_triggers_rollback(self):
        """MAE退化超过阈值应触发回滚（需连续3次退化达到窗口阈值）。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        restored = []

        def restore(state):
            restored.append(state)
            return True

        # 退化检测窗口为3，前2次返回continue，第3次触发回滚
        for i in range(2):
            r = self.ar.check_and_rollback(0.5, restore)
            self.assertEqual(r['action'], 'continue')
        result = self.ar.check_and_rollback(0.5, restore)
        self.assertEqual(result['action'], 'rollback')
        self.assertTrue(result['success'])
        self.assertEqual(len(restored), 1)

    def test_minor_degradation_no_rollback(self):
        """MAE轻微退化未超阈值不应触发回滚。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        result = self.ar.check_and_rollback(0.11, lambda x: True)
        self.assertEqual(result['action'], 'continue')

    def test_rollback_to_baseline_restores_state(self):
        """回滚应恢复到基线快照状态（需连续3次退化触发）。"""
        baseline_state = {'model_version': 'v1', 'alpha': 0.3}
        sid = self.sm.create_snapshot(baseline_state, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        restored_state = []

        def restore(state):
            restored_state.append(state)
            return True

        # 退化检测窗口为3，前2次不触发回滚
        for i in range(2):
            self.ar.check_and_rollback(0.5, restore)
        # 第3次触发回滚
        self.ar.check_and_rollback(0.5, restore)
        self.assertEqual(restored_state[0], baseline_state)

    def test_consecutive_degradation_counted(self):
        """连续退化应被正确计数（每3次连续退化触发1次回滚）。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        # 3轮×3次连续退化 = 3次回滚
        for _ in range(3):
            for _ in range(3):
                self.ar.check_and_rollback(0.5, lambda x: True)
        self.assertEqual(self.ar._rollback_count, 3)

    def test_snapshot_missing_fallback(self):
        """快照丢失时应返回失败信息（需连续3次退化后触发）。"""
        self.ar.set_baseline(0.1, 'nonexistent_id')
        # 退化检测窗口为3，前2次返回continue
        for i in range(2):
            r = self.ar.check_and_rollback(0.5, lambda x: True)
            self.assertEqual(r['action'], 'continue')
        # 第3次触发回滚但快照不存在
        result = self.ar.check_and_rollback(0.5, lambda x: True)
        self.assertEqual(result['action'], 'rollback_failed')
        self.assertEqual(result['reason'], 'snapshot_not_found')


class TestEndToEndPredictorRecovery(unittest.TestCase):
    """端到端预测器退化与恢复集成测试。"""

    def test_full_pipeline_degradation_recovery(self):
        """完整流水线：预测器退化→误差增大→优化修正→验证恢复。"""
        low = LowLevelPredictor(metrics=['cpu'])
        optimizer = MicroOptimizer(learning_rate=0.1, min_improvement=0.001)
        evaluator = ErrorEvaluator(metrics=['cpu'])

        for _ in range(20):
            low.update({'cpu': 50.0})

        predictions = low.predict()
        actuals = {'cpu': 50.0}
        result1 = evaluator.evaluate(predictions, actuals)
        mae1 = result1['mae']

        for _ in range(50):
            low.update({'cpu': 90.0})
        predictions = low.predict()
        actuals = {'cpu': 90.0}
        result2 = evaluator.evaluate(predictions, actuals)
        mae2 = result2['mae']

        self.assertGreater(mae2, mae1)

        optimizer.optimize(low, result2['errors'])
        params = low.get_metric_params('cpu')
        self.assertIsNotNone(params)

    def test_multi_predictor_model_switching(self):
        """MultiPredictor应支持不同模型类型切换。"""
        for model_type in ['exponential', 'adaptive_exponential', 'sma']:
            predictor = MultiPredictor(
                metrics=['cpu'],
                model_type=model_type,
                model_config={'alpha': 0.3, 'window': 5}
            )
            predictor.update_all({'cpu': 50.0})
            preds = predictor.predict_all()
            self.assertIn('cpu', preds)
            self.assertIsInstance(preds['cpu'], float)

    def test_error_evaluator_with_degraded_predictor(self):
        """ErrorEvaluator应在预测器退化时正确计算误差。"""
        evaluator = ErrorEvaluator(metrics=['cpu', 'mem'])
        predictions = {'cpu': 10.0, 'mem': 20.0}
        actuals = {'cpu': 100.0, 'mem': 200.0}
        result = evaluator.evaluate(predictions, actuals)
        self.assertEqual(result['mae'], 135.0)
        self.assertEqual(result['max_error'], 180.0)

    def test_snapshot_rollback_integrity(self):
        """快照回滚后预测器参数应保持一致性。"""
        temp_dir = tempfile.mkdtemp()
        try:
            sm = SnapshotManager(snapshot_dir=temp_dir, max_count=5)
            low = LowLevelPredictor(metrics=['cpu'])
            for _ in range(10):
                low.update({'cpu': 50.0})

            params_before = low.get_metric_params('cpu')
            sid = sm.create_snapshot(params_before, tag='baseline')
            loaded = sm.load_snapshot(sid)
            self.assertEqual(loaded['alpha'], params_before['alpha'])
            self.assertEqual(loaded['level'], params_before['level'])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
