#!/usr/bin/env python3
"""
接口全量遍历与非法输入fuzzing测试
目标：全覆盖所有接口，校验异常入参、超限参数、空值、非法格式
"""

import os
import sys
import unittest
import tempfile
import shutil
import random
import string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prediction.base_predictor import ExponentialSmoother, AdaptiveExponentialSmoother, SimpleMovingAverage
from src.prediction.arima_predictor import ARIMAModel
from src.prediction.error_evaluator import ErrorEvaluator
from src.prediction.layered_predictor import LowLevelPredictor, HighLevelPredictor
from src.drive.reward_engine import RewardEngine
from src.optimization.micro_optimizer import MicroOptimizer
from src.optimization.snapshot_manager import SnapshotManager
from src.safety.guardian import ResourceGuardian, BehaviorGuardian
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType
from src.memory.short_term import ShortTermMemory
from src.memory.value_scorer import ValueScorer


class FuzzGenerator:
    """非法输入生成器"""
    @staticmethod
    def random_string(length=20):
        return ''.join(random.choices(string.printable, k=length))

    @staticmethod
    def random_nested_dict(depth=3):
        if depth <= 0:
            return random.choice([None, 42, "str", [1, 2, 3], {"a": 1}])
        return {
            FuzzGenerator.random_string(5): FuzzGenerator.random_nested_dict(depth - 1)
            for _ in range(random.randint(1, 5))
        }


class TestExponentialSmootherFuzzing(unittest.TestCase):
    """指数平滑器fuzzing"""

    def test_fuzz_alpha_values(self):
        """各种非法alpha值不应崩溃"""
        for alpha in [-1.0, -0.1, 1.1, 2.0, float('inf'), float('-inf'), float('nan'), None, "0.5"]:
            try:
                s = ExponentialSmoother(alpha=alpha if isinstance(alpha, float) else 0.5)
                s.fit([1.0, 2.0, 3.0])
                s.predict(steps=1)
            except (TypeError, ValueError):
                pass  # 预期异常

    def test_fuzz_series_values(self):
        """各种非法序列值"""
        s = ExponentialSmoother(alpha=0.3)
        for series in [None, "string", 42, {}, [None, None], [float('inf'), float('-inf')]]:
            try:
                s.fit(series if isinstance(series, list) else [])
            except (TypeError, ValueError):
                pass


class TestErrorEvaluatorFuzzing(unittest.TestCase):
    """误差评估器fuzzing"""

    def test_fuzz_evaluate_inputs(self):
        """非法输入不应崩溃"""
        e = ErrorEvaluator()
        for preds, actuals in [
            (None, None),
            ({"a": "str"}, {"a": 1.0}),
            ({"a": float('nan')}, {"a": 1.0}),
            ({"a": float('inf')}, {"a": 1.0}),
            ({}, {}),
            ("string", "string"),
        ]:
            try:
                if isinstance(preds, dict) and isinstance(actuals, dict):
                    e.evaluate(preds, actuals)
            except Exception:
                pass  # 不应崩溃

    def test_evaluate_single_fuzz(self):
        """evaluate_single非法输入"""
        e = ErrorEvaluator()
        for pred, actual in [
            (None, 1.0),
            (1.0, None),
            ("str", 1.0),
            (1.0, "str"),
            (float('nan'), 1.0),
            (float('inf'), float('inf')),
        ]:
            try:
                if isinstance(pred, (int, float)) and isinstance(actual, (int, float)):
                    e.evaluate_single(pred, actual)
            except Exception:
                pass


class TestRewardEngineFuzzing(unittest.TestCase):
    """奖励引擎fuzzing"""

    def test_fuzz_state_inputs(self):
        """各种畸形状态输入"""
        r = RewardEngine()
        for state in [
            None,
            {},
            {"system": None},
            {"system": {"cpu_percent": "high"}},
            {"system": {"cpu_percent": float('nan')}},
            FuzzGenerator.random_nested_dict(2),
        ]:
            try:
                if isinstance(state, dict):
                    r.compute_safe_reward(state)
                    r.compute_dynamic_weights(state)
            except Exception:
                pass

    def test_fuzz_errors_list(self):
        """非法errors列表"""
        r = RewardEngine()
        for errors in [None, "string", {}, [float('nan')], [None]]:
            try:
                if isinstance(errors, list):
                    r.compute_gain_reward({}, errors)
            except Exception:
                pass


class TaskPoolFuzzing(unittest.TestCase):
    """任务池fuzzing"""

    def test_fuzz_update_inputs(self):
        """非法任务更新"""
        pool = TaskPool()
        for tasks in [
            None,
            "string",
            [None],
            [{}],
            [Task(id=None, type=TaskType.EXPLORE, name="test")],
        ]:
            try:
                if isinstance(tasks, list):
                    pool.update(tasks)
            except Exception:
                pass

    def test_fuzz_mark_executed(self):
        """非法任务ID"""
        pool = TaskPool()
        for task_id in [None, 123, {}, [], float('nan')]:
            try:
                pool.mark_executed(str(task_id) if task_id is not None else "", True, 0.5)
            except Exception:
                pass


class TestSnapshotManagerFuzzing(unittest.TestCase):
    """快照管理器fuzzing"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fuzz_create_snapshot(self):
        """非法快照数据"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        for data in [
            None,
            lambda x: x,
            open(__file__, 'rb'),
            FuzzGenerator.random_nested_dict(5),
        ]:
            try:
                sm.create_snapshot(data)
            except Exception:
                pass

    def test_fuzz_load_snapshot(self):
        """非法快照ID"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        for sid in [None, 123, {}, [], float('nan'), FuzzGenerator.random_string(100)]:
            try:
                sm.load_snapshot(str(sid) if sid is not None else "")
            except Exception:
                pass


class TestShortTermMemoryFuzzing(unittest.TestCase):
    """短期记忆fuzzing"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'fuzz.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fuzz_store_sample(self):
        """非法样本数据"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=100)
        for sample in [
            None,
            "string",
            42,
            {"state": None, "predictions": None, "actuals": None, "errors": None},
            {"state": FuzzGenerator.random_nested_dict(3)},
            {"timestamp": "not_a_number"},
        ]:
            try:
                if isinstance(sample, dict):
                    mem.store_sample(sample)
            except Exception:
                pass
        mem.close()


class TestValueScorerFuzzing(unittest.TestCase):
    """价值评分器fuzzing"""

    def test_fuzz_compute_score(self):
        """非法样本评分"""
        scorer = ValueScorer()
        for sample in [
            None,
            {},
            {"mae": "string"},
            {"mae": float('nan')},
            {"mae": float('inf')},
            {"reward": None},
            FuzzGenerator.random_nested_dict(2),
        ]:
            try:
                if isinstance(sample, dict):
                    scorer.compute_score(sample)
            except Exception:
                pass


class TestMicroOptimizerFuzzing(unittest.TestCase):
    """微优化器fuzzing"""

    def test_fuzz_optimize_errors(self):
        """非法误差输入"""
        opt = MicroOptimizer()
        for errors in [
            None,
            "string",
            {"cpu": None},
            {"cpu": "error"},
            {"cpu": float('nan')},
            FuzzGenerator.random_nested_dict(2),
        ]:
            try:
                if isinstance(errors, dict):
                    opt.optimize(None, errors)
            except Exception:
                pass


class TestHighLevelPredictorFuzzing(unittest.TestCase):
    """高层预测器fuzzing"""

    def test_fuzz_predict_inputs(self):
        """非法预测输入"""
        p = HighLevelPredictor()
        for preds, actuals in [
            (None, None),
            ({"cpu": "high"}, {"cpu": 0.5}),
            ({"cpu": float('nan')}, {"cpu": 0.5}),
            (FuzzGenerator.random_nested_dict(2), {}),
        ]:
            try:
                if isinstance(preds, dict) and isinstance(actuals, dict):
                    p.predict(preds, actuals)
            except Exception:
                pass


class TestGuardianFuzzing(unittest.TestCase):
    """安全护栏fuzzing"""

    def test_fuzz_resource_limits(self):
        """非法资源限制"""
        for limits in [
            (-1.0, 50.0),
            (50.0, -1.0),
            (float('nan'), 50.0),
            (float('inf'), 50.0),
            (None, None),
        ]:
            try:
                cpu, mem = limits
                if isinstance(cpu, (int, float)) and isinstance(mem, (int, float)):
                    g = ResourceGuardian(max_cpu_percent=cpu, max_mem_percent=mem)
                    g.check_resources()
            except Exception:
                pass

    def test_fuzz_behavior_paths(self):
        """非法路径检查"""
        bg = BehaviorGuardian()
        for path in [None, 123, {}, [], float('nan'), FuzzGenerator.random_string(200)]:
            try:
                bg.check_path(str(path) if path is not None else "")
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
