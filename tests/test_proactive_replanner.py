"""阶段150：主动式重规划与退化检测测试。"""

import unittest
from unittest.mock import MagicMock

from src.planning.proactive_replanner import DegradationDetector, ProactiveReplanner
from src.task.task_definition import Task


class TestDegradationDetector(unittest.TestCase):
    """DegradationDetector 退化检测测试。"""

    def test_no_degradation_on_empty(self):
        """无观察数据时不视为退化。"""
        detector = DegradationDetector()
        self.assertFalse(detector.is_degraded())

    def test_no_degradation_on_success(self):
        """全部成功不视为退化。"""
        detector = DegradationDetector(window_size=4)
        for _ in range(4):
            detector.observe({'success': True, 'duration': 0.1})
        self.assertFalse(detector.is_degraded())

    def test_consecutive_failures_trigger_degradation(self):
        """连续失败超过阈值触发退化。"""
        detector = DegradationDetector(
            window_size=4,
            consecutive_failure_threshold=3,
        )
        for _ in range(3):
            detector.observe({'success': False, 'duration': 0.1})
        self.assertTrue(detector.is_degraded())
        signals = detector.get_signals()
        self.assertIn('consecutive_failures', signals['reasons'])

    def test_low_success_rate_trigger_degradation(self):
        """成功率低于阈值触发退化。"""
        detector = DegradationDetector(
            window_size=4,
            success_rate_threshold=0.5,
            consecutive_failure_threshold=100,  # 禁用连续失败检测
        )
        # 3 失败 1 成功 = 25% 成功率
        detector.observe({'success': True, 'duration': 0.1})
        for _ in range(3):
            detector.observe({'success': False, 'duration': 0.1})
        self.assertTrue(detector.is_degraded())
        signals = detector.get_signals()
        self.assertIn('low_success_rate', signals['reasons'])

    def test_latency_growth_trigger_degradation(self):
        """延迟增长率超过阈值触发退化。"""
        detector = DegradationDetector(
            window_size=4,
            latency_growth_threshold=2.0,
            consecutive_failure_threshold=100,
            success_rate_threshold=0.0,
        )
        # 前半段 0.1s，后半段 0.5s -> 增长率 5x
        detector.observe({'success': True, 'duration': 0.1})
        detector.observe({'success': True, 'duration': 0.1})
        detector.observe({'success': True, 'duration': 0.5})
        detector.observe({'success': True, 'duration': 0.5})
        self.assertTrue(detector.is_degraded())
        signals = detector.get_signals()
        self.assertIn('latency_growth', signals['reasons'])

    def test_no_degradation_with_stable_latency(self):
        """延迟稳定时不触发退化。"""
        detector = DegradationDetector(
            window_size=4,
            latency_growth_threshold=2.0,
            consecutive_failure_threshold=100,
            success_rate_threshold=0.0,
        )
        for _ in range(4):
            detector.observe({'success': True, 'duration': 0.1})
        self.assertFalse(detector.is_degraded())

    def test_reset_clears_state(self):
        """reset 清空观察数据。"""
        detector = DegradationDetector(consecutive_failure_threshold=2)
        detector.observe({'success': False, 'duration': 0.1})
        detector.observe({'success': False, 'duration': 0.1})
        self.assertTrue(detector.is_degraded())
        detector.reset()
        self.assertFalse(detector.is_degraded())

    def test_get_signals_returns_details(self):
        """get_signals 返回详细信号。"""
        detector = DegradationDetector(window_size=4)
        detector.observe({'success': True, 'duration': 0.1})
        detector.observe({'success': False, 'duration': 0.2})
        signals = detector.get_signals()
        self.assertIn('window_size', signals)
        self.assertIn('consecutive_failures', signals)
        self.assertIn('latency_growth', signals)
        self.assertIn('recent_success_rate', signals)
        self.assertIn('is_degraded', signals)
        self.assertIn('reasons', signals)

    def test_success_resets_consecutive_failures(self):
        """成功执行重置连续失败计数。"""
        detector = DegradationDetector(consecutive_failure_threshold=3)
        detector.observe({'success': False, 'duration': 0.1})
        detector.observe({'success': False, 'duration': 0.1})
        detector.observe({'success': True, 'duration': 0.1})
        # 连续失败被重置，不应触发退化
        self.assertFalse(detector._check_consecutive_failures())

    def test_partial_window_no_success_rate_degradation(self):
        """窗口未填满时不触发成功率退化。"""
        detector = DegradationDetector(
            window_size=6,
            success_rate_threshold=0.5,
            consecutive_failure_threshold=100,
        )
        for _ in range(3):
            detector.observe({'success': False, 'duration': 0.1})
        # 窗口未填满，不检测成功率
        self.assertFalse(detector.is_degraded())


class TestProactiveReplanner(unittest.TestCase):
    """ProactiveReplanner 主动重规划测试。"""

    def setUp(self):
        self.detector = DegradationDetector(
            window_size=4,
            consecutive_failure_threshold=2,
        )
        self.replan_callback = MagicMock(return_value=None)
        self.replanner = ProactiveReplanner(
            self.detector,
            self.replan_callback,
            cooldown=2,
        )

    def test_no_replan_before_cooldown(self):
        """冷却期内不触发重规划。"""
        self.replanner.observe({'success': False, 'duration': 0.1})
        # 冷却期=2，仅观察1步，不应触发
        result = self.replanner.check(planner=None, current_index=0)
        self.assertIsNone(result)
        self.replan_callback.assert_not_called()

    def test_no_replan_without_degradation(self):
        """无退化时不触发重规划。"""
        self.replanner.observe({'success': True, 'duration': 0.1})
        self.replanner.observe({'success': True, 'duration': 0.1})
        result = self.replanner.check(planner=None, current_index=0)
        self.assertIsNone(result)
        self.replan_callback.assert_not_called()

    def test_replan_on_degradation(self):
        """退化时触发重规划。"""
        # 观察足够多的失败以触发退化
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replanner.observe({'success': False, 'duration': 0.1})
        # 冷却期已过且检测到退化
        new_steps = [Task(name='recovery')]
        self.replan_callback.return_value = new_steps
        result = self.replanner.check(planner=None, current_index=0)
        self.assertEqual(result, new_steps)
        self.replan_callback.assert_called_once()
        self.assertEqual(self.replanner.replan_count, 1)

    def test_replan_resets_detector(self):
        """重规划后检测器被重置。"""
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replan_callback.return_value = [Task(name='recovery')]
        self.replanner.check(planner=None, current_index=0)
        # 检测器应被重置
        self.assertFalse(self.detector.is_degraded())

    def test_replan_resets_cooldown(self):
        """重规划后冷却期重新计算。"""
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replan_callback.return_value = [Task(name='recovery')]
        self.replanner.check(planner=None, current_index=0)

        # 再次观察失败但冷却期未过
        self.replanner.observe({'success': False, 'duration': 0.1})
        result = self.replanner.check(planner=None, current_index=0)
        self.assertIsNone(result)

    def test_replan_callback_returns_none(self):
        """重规划回调返回 None 时不计数。"""
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replan_callback.return_value = None
        result = self.replanner.check(planner=None, current_index=0)
        self.assertIsNone(result)
        self.assertEqual(self.replanner.replan_count, 0)

    def test_reset_clears_state(self):
        """reset 清空全部状态。"""
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replanner.reset()
        self.assertEqual(self.replanner.replan_count, 0)
        self.assertFalse(self.detector.is_degraded())

    def test_signals_passed_to_callback(self):
        """退化信号应传递给重规划回调。"""
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replanner.observe({'success': False, 'duration': 0.1})
        self.replan_callback.return_value = [Task(name='recovery')]
        self.replanner.check(planner='fake_planner', current_index=5)

        call_args = self.replan_callback.call_args
        self.assertEqual(call_args[0][0], 'fake_planner')
        self.assertEqual(call_args[0][1], 5)
        signals = call_args[0][2]
        self.assertTrue(signals['is_degraded'])


if __name__ == '__main__':
    unittest.main()
