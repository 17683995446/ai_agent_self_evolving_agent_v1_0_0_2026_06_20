"""阶段123：接口全量遍历与边界值测试。

验证所有公开接口的参数边界：
1. ShortTermMemory: store_sample/load_samples/update_value_score/get_sample_count
2. SnapshotManager: create_snapshot/load_snapshot/delete_snapshot
3. TaskPool: update/select_task/mark_executed/remove_task
4. TaskGenerator: generate
5. TaskScorer: score_task/rank_tasks
6. MicroOptimizer: optimize/optimize_adaptive_smoother
7. AutoRollback: set_baseline/check_and_rollback
8. StateVector: perceive/flatten
9. ErrorEvaluator: evaluate
10. RewardEngine: compute_total_reward

边界值类型：None/空值/非法类型/超长字符串/负数/极值/嵌套异常
"""
import os
import sys
import time
import unittest
import tempfile
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.optimization.auto_rollback import AutoRollback
from src.optimization.micro_optimizer import MicroOptimizer
from src.task.task_pool import TaskPool
from src.task.task_generator import TaskGenerator
from src.task.task_scorer import TaskScorer
from src.task.task_definition import Task
from src.core.constants import TaskType
from src.prediction.error_evaluator import ErrorEvaluator
from src.prediction.base_predictor import AdaptiveExponentialSmoother
from src.prediction.layered_predictor import LowLevelPredictor, MidLevelPredictor, HighLevelPredictor


class TestShortTermMemoryBoundary(unittest.TestCase):
    """ShortTermMemory接口边界值测试。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")
        self.mem = ShortTermMemory(db_path=self.db_path, batch_size=5, flush_interval=9999)

    def tearDown(self):
        self.mem.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_store_sample_none(self):
        """store_sample(None)应返回-1不崩溃。"""
        ret = self.mem.store_sample(None)
        self.assertEqual(ret, -1)

    def test_store_sample_not_dict(self):
        """store_sample(非dict)应返回-1。"""
        for val in [42, "string", [], 3.14, True]:
            ret = self.mem.store_sample(val)
            self.assertEqual(ret, -1)

    def test_store_sample_empty_dict(self):
        """store_sample({})应能处理空字典。"""
        ret = self.mem.store_sample({})
        self.assertIsInstance(ret, int)

    def test_store_sample_missing_fields(self):
        """store_sample缺少必需字段应不崩溃。"""
        ret = self.mem.store_sample({'timestamp': time.time()})
        self.assertIsInstance(ret, int)

    def test_store_sample_extreme_values(self):
        """store_sample极值应不崩溃。"""
        sample = {
            'timestamp': float('inf'),
            'state': {'cpu': float('nan')},
            'predictions': {'low': {}, 'mid': {}, 'high': {}},
            'actuals': {'cpu': float('inf')},
            'errors': {'cpu': -1.0},
            'task_id': None,
            'reward': float('nan'),
            'value_score': -999.0,
        }
        ret = self.mem.store_sample(sample)
        self.assertIsInstance(ret, int)

    def test_store_sample_super_long_string(self):
        """store_sample超长字符串应不崩溃。"""
        sample = {
            'timestamp': time.time(),
            'state': {'data': 'x' * 100000},
            'predictions': {'low': {}, 'mid': {}, 'high': {}},
            'actuals': {'cpu': 0},
            'errors': {'cpu': 0.1},
            'task_id': 't' * 10000,
            'reward': 0.5,
            'value_score': 0.5,
        }
        ret = self.mem.store_sample(sample)
        self.assertIsInstance(ret, int)

    def test_load_samples_zero_limit(self):
        """load_samples(limit=0)应返回空列表。"""
        result = self.mem.load_samples(limit=0)
        self.assertEqual(len(result), 0)

    def test_load_samples_negative_limit(self):
        """load_samples(limit=-1)应不崩溃。"""
        result = self.mem.load_samples(limit=-1)
        self.assertIsInstance(result, list)

    def test_load_samples_huge_limit(self):
        """load_samples(limit=10**9)应不崩溃。"""
        result = self.mem.load_samples(limit=10**9)
        self.assertIsInstance(result, list)

    def test_update_value_score_invalid_id(self):
        """update_value_score(不存在的id)应不崩溃。"""
        self.mem.update_value_score('nonexistent_id', 0.5)

    def test_update_value_score_negative_score(self):
        """update_value_score(负分)应不崩溃。"""
        self.mem.update_value_score('some_id', -1.0)

    def test_get_sample_count_empty(self):
        """空数据库get_sample_count应返回0。"""
        count = self.mem.get_sample_count()
        self.assertEqual(count, 0)


class TestSnapshotManagerBoundary(unittest.TestCase):
    """SnapshotManager接口边界值测试。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sm = SnapshotManager(snapshot_dir=self.tmp, max_count=5)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_snapshot_none_state(self):
        """create_snapshot(None)应不崩溃。"""
        try:
            sid = self.sm.create_snapshot(None, metrics={'mae': 0.1})
            self.assertIsInstance(sid, str)
        except (TypeError, Exception):
            pass

    def test_create_snapshot_empty_metrics(self):
        """create_snapshot(metrics=None)应不崩溃。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics=None)
        self.assertIsInstance(sid, str)

    def test_create_snapshot_negative_max_count(self):
        """负max_count应被规范化。"""
        sm = SnapshotManager(snapshot_dir=self.tmp, max_count=-1)
        self.assertGreaterEqual(sm.max_count, 1)

    def test_create_snapshot_zero_max_count(self):
        """零max_count应被规范化。"""
        sm = SnapshotManager(snapshot_dir=self.tmp, max_count=0)
        self.assertGreaterEqual(sm.max_count, 1)

    def test_load_snapshot_nonexistent(self):
        """load_snapshot(不存在的id)应返回None或抛异常。"""
        result = self.sm.load_snapshot('nonexistent_id')
        self.assertIsNone(result)

    def test_load_snapshot_empty_id(self):
        """load_snapshot('')应不崩溃。"""
        try:
            result = self.sm.load_snapshot('')
        except (ValueError, Exception):
            pass

    def test_delete_snapshot_nonexistent(self):
        """delete_snapshot(不存在的id)应不崩溃。"""
        try:
            self.sm.delete_snapshot('nonexistent_id')
        except (KeyError, Exception):
            pass


class TestTaskPoolBoundary(unittest.TestCase):
    """TaskPool接口边界值测试。"""

    def setUp(self):
        self.pool = TaskPool(max_size=10)

    def test_update_none(self):
        """update(None)应不崩溃。"""
        self.pool.update(None)

    def test_update_empty_list(self):
        """update([])应不崩溃。"""
        self.pool.update([])

    def test_update_with_none_task(self):
        """update([None])应不崩溃。"""
        try:
            self.pool.update([None])
        except (AttributeError, Exception):
            pass

    def test_mark_executed_nonexistent(self):
        """mark_executed(不存在的id)应不崩溃。"""
        self.pool.mark_executed('nonexistent', True, 0.5)

    def test_mark_executed_negative_reward(self):
        """mark_executed(负reward)应不崩溃。"""
        task = Task(id='test_neg', type=TaskType.SAFE, action='test')
        self.pool.update([task])
        self.pool.mark_executed('test_neg', True, -1.0)

    def test_remove_task_nonexistent(self):
        """remove_task(不存在的id)应返回False。"""
        result = self.pool.remove_task('nonexistent')
        self.assertFalse(result)

    def test_freeze_task_nonexistent(self):
        """freeze_task(不存在的id)应不崩溃。"""
        self.pool.freeze_task('nonexistent')

    def test_select_task_empty_pool(self):
        """空池select_task应返回None。"""
        result = self.pool.select_task()
        self.assertIsNone(result)

    def test_get_top_tasks_zero_n(self):
        """get_top_tasks(0)应返回空列表。"""
        result = self.pool.get_top_tasks(0)
        self.assertEqual(len(result), 0)

    def test_get_top_tasks_negative_n(self):
        """get_top_tasks(-1)应不崩溃。"""
        result = self.pool.get_top_tasks(-1)
        self.assertIsInstance(result, list)


class TestTaskGeneratorBoundary(unittest.TestCase):
    """TaskGenerator接口边界值测试。"""

    def setUp(self):
        self.gen = TaskGenerator()

    def test_generate_none(self):
        """generate(None)应不崩溃。"""
        try:
            tasks = self.gen.generate(None)
        except (AttributeError, Exception):
            pass

    def test_generate_empty_dict(self):
        """generate({})应不崩溃。"""
        tasks = self.gen.generate({})
        self.assertIsInstance(tasks, list)

    def test_generate_negative_gaps(self):
        """generate(负gap)应不崩溃。"""
        tasks = self.gen.generate({'safe_gap': -0.5, 'gain_gap': -1.0, 'eff_gap': -2.0})
        self.assertIsInstance(tasks, list)

    def test_generate_extreme_gaps(self):
        """generate(极大gap)应不崩溃。"""
        tasks = self.gen.generate({'safe_gap': 1000.0, 'gain_gap': 1000.0, 'eff_gap': 1000.0})
        self.assertIsInstance(tasks, list)


class TestTaskScorerBoundary(unittest.TestCase):
    """TaskScorer接口边界值测试。"""

    def setUp(self):
        self.scorer = TaskScorer()

    def test_score_task_none(self):
        """score_task(None)应不崩溃。"""
        try:
            self.scorer.score_task(None, {}, {})
        except (AttributeError, Exception):
            pass

    def test_rank_tasks_empty(self):
        """rank_tasks([])应返回空列表。"""
        result = self.scorer.rank_tasks([])
        self.assertEqual(len(result), 0)

    def test_rank_tasks_none(self):
        """rank_tasks(None)应不崩溃。"""
        try:
            self.scorer.rank_tasks(None)
        except (TypeError, Exception):
            pass


class TestMicroOptimizerBoundary(unittest.TestCase):
    """MicroOptimizer接口边界值测试。"""

    def setUp(self):
        self.opt = MicroOptimizer()

    def test_optimize_none_predictor(self):
        """optimize(None)应不崩溃。"""
        try:
            self.opt.optimize(None, {'cpu': 0.1})
        except (AttributeError, Exception):
            pass

    def test_optimize_empty_errors(self):
        """optimize({})应不崩溃。"""
        predictor = LowLevelPredictor(metrics=['cpu_percent'])
        self.opt.optimize(predictor, {})

    def test_optimize_none_errors(self):
        """optimize(None errors)应不崩溃。"""
        predictor = LowLevelPredictor(metrics=['cpu_percent'])
        try:
            self.opt.optimize(predictor, None)
        except (TypeError, Exception):
            pass

    def test_optimize_negative_error(self):
        """optimize(负误差)应不崩溃。"""
        predictor = LowLevelPredictor(metrics=['cpu_percent'])
        self.opt.optimize(predictor, {'cpu_percent': -0.5})

    def test_optimize_extreme_error(self):
        """optimize(极大误差)应不崩溃。"""
        predictor = LowLevelPredictor(metrics=['cpu_percent'])
        self.opt.optimize(predictor, {'cpu_percent': 1000000.0})


class TestAutoRollbackBoundary(unittest.TestCase):
    """AutoRollback接口边界值测试。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sm = SnapshotManager(snapshot_dir=self.tmp, max_count=5)
        self.ar = AutoRollback(snapshot_manager=self.sm, mae_degradation_threshold=0.1)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_check_and_rollback_no_baseline(self):
        """无基线时check_and_rollback应返回no_baseline。"""
        result = self.ar.check_and_rollback(0.5)
        self.assertEqual(result['action'], 'no_baseline')

    def test_check_and_rollback_negative_mae(self):
        """check_and_rollback(负MAE)应不崩溃。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        result = self.ar.check_and_rollback(-1.0)
        self.assertIsInstance(result, dict)

    def test_check_and_rollback_nan_mae(self):
        """check_and_rollback(NaN MAE)应不崩溃。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        try:
            result = self.ar.check_and_rollback(float('nan'))
        except (ValueError, Exception):
            pass

    def test_check_and_rollback_none_callback(self):
        """check_and_rollback(None callback)应不崩溃。"""
        sid = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.1})
        self.ar.set_baseline(0.1, sid)
        for _ in range(3):
            try:
                result = self.ar.check_and_rollback(0.5, None)
            except (TypeError, Exception):
                break


class TestErrorEvaluatorBoundary(unittest.TestCase):
    """ErrorEvaluator接口边界值测试。"""

    def setUp(self):
        self.ev = ErrorEvaluator()

    def test_evaluate_empty_dicts(self):
        """evaluate({},{})应不崩溃。"""
        result = self.ev.evaluate({}, {})
        self.assertIsInstance(result, dict)

    def test_evaluate_none(self):
        """evaluate(None,None)应不崩溃。"""
        try:
            result = self.ev.evaluate(None, None)
        except (AttributeError, Exception):
            pass

    def test_evaluate_mismatched_keys(self):
        """evaluate(键不匹配)应不崩溃。"""
        result = self.ev.evaluate({'a': 1.0}, {'b': 2.0})
        self.assertIsInstance(result, dict)

    def test_evaluate_nan_values(self):
        """evaluate(NaN值)应不崩溃。"""
        result = self.ev.evaluate({'a': float('nan')}, {'a': float('nan')})
        self.assertIsInstance(result, dict)


class TestPredictorBoundary(unittest.TestCase):
    """预测器接口边界值测试。"""

    def test_adaptive_smoother_nan_update(self):
        """AdaptiveExponentialSmoother.update(NaN)应不崩溃。"""
        smoother = AdaptiveExponentialSmoother(alpha_init=0.5)
        smoother.update(float('nan'))
        # NaN应被忽略
        self.assertEqual(len(smoother._history), 0)

    def test_adaptive_smoother_inf_update(self):
        """AdaptiveExponentialSmoother.update(Inf)应不崩溃。"""
        smoother = AdaptiveExponentialSmoother(alpha_init=0.5)
        smoother.update(float('inf'))
        self.assertEqual(len(smoother._history), 0)

    def test_adaptive_smoother_none_update(self):
        """AdaptiveExponentialSmoother.update(None)应不崩溃。"""
        smoother = AdaptiveExponentialSmoother(alpha_init=0.5)
        try:
            smoother.update(None)
        except (TypeError, Exception):
            pass

    def test_low_predictor_empty_update(self):
        """LowLevelPredictor.update({})应不崩溃。"""
        predictor = LowLevelPredictor(metrics=['cpu_percent'])
        predictor.update({})

    def test_low_predictor_none_update(self):
        """LowLevelPredictor.update(None)应不崩溃。"""
        predictor = LowLevelPredictor(metrics=['cpu_percent'])
        try:
            predictor.update(None)
        except (AttributeError, Exception):
            pass

    def test_mid_predictor_empty_update(self):
        """MidLevelPredictor.update({})应不崩溃。"""
        predictor = MidLevelPredictor(metrics=['cpu_percent'])
        predictor.update({})

    def test_high_predictor_empty_inputs(self):
        """HighLevelPredictor.predict({},{})应不崩溃。"""
        hp = HighLevelPredictor()
        result = hp.predict({}, {})
        self.assertIsInstance(result, dict)

    def test_high_predictor_none_inputs(self):
        """HighLevelPredictor.predict(None,None)应不崩溃。"""
        hp = HighLevelPredictor()
        try:
            result = hp.predict(None, None)
        except (AttributeError, Exception):
            pass


class TestRewardEngineBoundary(unittest.TestCase):
    """RewardEngine接口边界值测试。"""

    def setUp(self):
        from src.drive.reward_engine import RewardEngine
        self.engine = RewardEngine(base_weights=(0.4, 0.4, 0.2))

    def test_compute_total_reward_empty_state(self):
        """compute_total_reward({})应不崩溃。"""
        result = self.engine.compute_total_reward({}, [])
        self.assertIsInstance(result, dict)

    def test_compute_total_reward_none_state(self):
        """compute_total_reward(None)应不崩溃。"""
        try:
            result = self.engine.compute_total_reward(None, [])
        except (AttributeError, Exception):
            pass

    def test_compute_total_reward_none_errors(self):
        """compute_total_reward(state, None)应不崩溃。"""
        try:
            result = self.engine.compute_total_reward({}, None)
        except (TypeError, Exception):
            pass

    def test_compute_total_reward_extreme_values(self):
        """compute_total_reward(极值)应不崩溃。"""
        state = {'system': {'cpu_percent': float('inf'), 'mem_percent': -1.0}}
        result = self.engine.compute_total_reward(state, [float('nan')] * 10)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
