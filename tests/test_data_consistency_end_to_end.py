import os
import sys
import unittest
import tempfile
import shutil
import json
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.short_term import ShortTermMemory
from src.prediction.layered_predictor import LowLevelPredictor, MidLevelPredictor, HighLevelPredictor
from src.prediction.error_evaluator import ErrorEvaluator
from src.optimization.snapshot_manager import SnapshotManager
from src.safety.guardian import ResourceGuardian, BehaviorGuardian, SafetyEnforcer
from src.task.task_pool import TaskPool
from src.task.task_definition import Task
from src.core.constants import TaskType


class TestShortTermMemoryConsistency(unittest.TestCase):
    """短期记忆读写一致性验证。"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'memory.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_load_roundtrip(self):
        """存储后加载的数据应与原始数据一致。"""
        memory = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        sample = {
            'timestamp': 1234567890.0,
            'state': {'cpu': 50.0},
            'predictions': {'low': {'cpu': 48.0}},
            'actuals': {'cpu': 50.0},
            'errors': {'cpu': 2.0},
            'task_id': 'task_1',
            'reward': 0.8,
            'value_score': 0.9,
        }
        row_id = memory.store_sample(sample)
        self.assertGreater(row_id, 0)

        loaded = memory.load_samples(limit=1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['state'], sample['state'])
        self.assertEqual(loaded[0]['predictions'], sample['predictions'])
        self.assertEqual(loaded[0]['actuals'], sample['actuals'])
        self.assertEqual(loaded[0]['errors'], sample['errors'])
        self.assertEqual(loaded[0]['task_id'], sample['task_id'])
        self.assertAlmostEqual(loaded[0]['reward'], sample['reward'], places=5)

    def test_mae_computed_correctly(self):
        """加载时应正确计算mae字段。"""
        memory = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        sample = {
            'timestamp': time.time(),
            'state': {},
            'predictions': {},
            'actuals': {},
            'errors': {'cpu': 4.0, 'mem': 6.0},
            'task_id': None,
            'reward': 0.0,
            'value_score': 0.0,
        }
        memory.store_sample(sample)
        loaded = memory.load_samples(limit=1)
        expected_mae = (4.0 + 6.0) / 2.0
        self.assertAlmostEqual(loaded[0]['mae'], expected_mae, places=5)

    def test_empty_errors_mae_zero(self):
        """空误差时mae应为0。"""
        memory = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        sample = {
            'timestamp': time.time(),
            'state': {},
            'predictions': {},
            'actuals': {},
            'errors': {},
            'task_id': None,
            'reward': 0.0,
            'value_score': 0.0,
        }
        memory.store_sample(sample)
        loaded = memory.load_samples(limit=1)
        self.assertEqual(loaded[0]['mae'], 0.0)

    def test_sample_count_accuracy(self):
        """样本计数应准确。"""
        memory = ShortTermMemory(db_path=self.db_path, max_samples=1000, batch_size=1, flush_interval=0.0)
        for i in range(50):
            memory.store_sample({
                'timestamp': float(i),
                'state': {'idx': i},
                'predictions': {},
                'actuals': {},
                'errors': {},
                'task_id': None,
                'reward': 0.0,
                'value_score': float(i),
            })
        self.assertEqual(memory.get_sample_count(), 50)

    def test_cleanup_retains_high_value(self):
        """清理后高价值样本应被保留。"""
        memory = ShortTermMemory(db_path=self.db_path, max_samples=10, batch_size=1, flush_interval=0.0)
        for i in range(20):
            memory.store_sample({
                'timestamp': float(i),
                'state': {},
                'predictions': {},
                'actuals': {},
                'errors': {},
                'task_id': None,
                'reward': 0.0,
                'value_score': float(i),
            })
        memory.cleanup_old_samples(retention_ratio=0.5)
        count = memory.get_sample_count()
        self.assertEqual(count, 10)

        loaded = memory.load_high_value_samples(limit=10)
        scores = [s['value_score'] for s in loaded]
        self.assertEqual(sorted(scores, reverse=True), scores)

    def test_concurrent_store_consistency(self):
        """并发存储不应丢失数据或损坏数据库。"""
        memory = ShortTermMemory(db_path=self.db_path, max_samples=10000, batch_size=1, flush_interval=0.0)
        errors = []

        def store_batch(start):
            try:
                for i in range(10):
                    memory.store_sample({
                        'timestamp': float(start + i),
                        'state': {'batch': start},
                        'predictions': {},
                        'actuals': {},
                        'errors': {},
                        'task_id': None,
                        'reward': 0.0,
                        'value_score': 0.0,
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_batch, args=(i * 10,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(memory.get_sample_count(), 100)


class TestPredictorStateConsistency(unittest.TestCase):
    """预测器状态更新与预测一致性验证。"""

    def test_low_level_update_predict_consistency(self):
        """底层预测器update后predict应反映最新数据。"""
        low = LowLevelPredictor(metrics=['cpu'])
        low.update({'cpu': 50.0})
        pred1 = low.predict()['cpu']
        low.update({'cpu': 60.0})
        pred2 = low.predict()['cpu']
        self.assertNotEqual(pred1, pred2)
        self.assertGreater(pred2, pred1)

    def test_mid_level_history_matches_updates(self):
        """中层预测器历史应与update次数一致。"""
        mid = MidLevelPredictor(metrics=['cpu'])
        for i in range(10):
            mid.update({'cpu': float(i)})
        history = mid.predictors['cpu']._history
        self.assertEqual(len(history), 10)

    def test_high_level_strategy_scores_monotonic(self):
        """高层策略评分在持续正向反馈下应单调不减。"""
        high = HighLevelPredictor()
        scores = []
        for _ in range(5):
            high.update({'system': {'cpu_percent': 50.0}}, 'explore', 1.0)
            scores.append(high._strategy_scores['explore'])
        for i in range(1, len(scores)):
            self.assertGreaterEqual(scores[i], scores[i - 1])

    def test_predictor_params_serialization_roundtrip(self):
        """预测器参数序列化反序列化应保持一致。"""
        low = LowLevelPredictor(metrics=['cpu'])
        low.update({'cpu': 50.0})
        params = low.get_metric_params('cpu')

        low2 = LowLevelPredictor(metrics=['cpu'])
        low2.set_metric_params('cpu', params)
        pred1 = low.predict()['cpu']
        pred2 = low2.predict()['cpu']
        self.assertAlmostEqual(pred1, pred2, places=5)


class TestSnapshotManagerConsistency(unittest.TestCase):
    """快照管理器序列化一致性验证。"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=5)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_snapshot_roundtrip(self):
        """快照创建后加载应与原始状态一致。"""
        state = {
            'cycle': 100,
            'errors': [0.1, 0.2, 0.3],
            'nested': {'a': 1, 'b': [1, 2, 3]},
        }
        sid = self.sm.create_snapshot(state, tag='test')
        loaded = self.sm.load_snapshot(sid)
        self.assertEqual(loaded, state)

    def test_snapshot_index_persistence(self):
        """索引应在重建后保持一致。"""
        self.sm.create_snapshot({'v': 1}, tag='a')
        self.sm.create_snapshot({'v': 2}, tag='b')

        sm2 = SnapshotManager(snapshot_dir=self.temp_dir, max_count=5)
        self.assertEqual(len(sm2.list_snapshots()), 2)

    def test_snapshot_metrics_integrity(self):
        """快照关联的metrics应完整保存。"""
        metrics = {'mae': 0.15, 'coverage': 0.95, 'eff_score': 0.8}
        sid = self.sm.create_snapshot({'v': 1}, tag='test', metrics=metrics)
        entry = next(s for s in self.sm.list_snapshots() if s['id'] == sid)
        self.assertEqual(entry['metrics'], metrics)

    def test_compare_snapshot_diffs(self):
        """快照比较应准确反映metrics差异。"""
        sid1 = self.sm.create_snapshot({'v': 1}, metrics={'mae': 0.2})
        time.sleep(0.01)
        sid2 = self.sm.create_snapshot({'v': 2}, metrics={'mae': 0.1})
        result = self.sm.compare_snapshots(sid1, sid2)
        self.assertEqual(result['diffs']['mae'], -0.1)


class TestSafetyEnforcerConsistency(unittest.TestCase):
    """安全执行器状态一致性验证。"""

    def test_emergency_state_reflects_resource_check(self):
        """紧急模式状态应准确反映资源检查结果。"""
        rg = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        result = se.full_check()
        self.assertTrue(result['emergency_mode'])
        self.assertFalse(result['resource_safe'])

    def test_clear_emergency_resets_state(self):
        """清除紧急模式后状态应重置。"""
        rg = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        se.full_check()
        self.assertTrue(se.is_emergency())
        se.clear_emergency()
        self.assertFalse(se.is_emergency())
        self.assertFalse(se._emergency_mode)


class TestTaskPoolConsistency(unittest.TestCase):
    """任务池数据一致性验证。"""

    def test_task_insert_and_retrieve(self):
        """任务插入后应能正确检索。"""
        pool = TaskPool(max_size=10)
        task = Task(id='t1', name='test_task', type=TaskType.EXPLORE)
        pool.update([task])
        self.assertIn('t1', pool.tasks)
        self.assertEqual(pool.tasks['t1'].name, 'test_task')

    def test_task_score_update_consistency(self):
        """任务评分更新应保持一致。"""
        pool = TaskPool(max_size=10)
        task = Task(id='t1', name='test_task', type=TaskType.EXPLORE)
        pool.update([task])
        pool.tasks['t1'].score = 0.8
        self.assertEqual(pool.tasks['t1'].score, 0.8)

    def test_freeze_task_persists(self):
        """任务冻结状态应持久。"""
        pool = TaskPool(max_size=10)
        task = Task(id='t1', name='test_task', type=TaskType.EXPLORE)
        pool.update([task])
        pool.freeze_task('t1')
        self.assertTrue(pool.tasks['t1'].frozen)


class TestEndToEndDataFlowConsistency(unittest.TestCase):
    """端到端数据流一致性验证。"""

    def test_agent_loop_result_structure(self):
        """AgentLoop.run_cycle结果应包含所有预期字段。"""
        from src.core.agent_loop import AgentLoop
        import tempfile, os
        temp_dir = tempfile.mkdtemp()
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)
        for d in ['data', 'logs', 'snapshots']:
            os.makedirs(d, exist_ok=True)
        with open('config.yaml', 'w') as f:
            f.write("""
goals:
  precision_evolution: true
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: 100.0
  max_mem_percent: 100.0
exploration:
  base_ratio: 0.3
safety:
  max_consecutive_failures: 5
review:
  interval_hours: 999
  min_samples: 99999
memory:
  max_short_term_samples: 100
logging:
  level: INFO
  file: logs/agent.log
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
dashboard:
  enabled: false
""")
        agent = AgentLoop(config_path='config.yaml')
        result = agent.run_cycle()
        self.assertIn('cycle', result)
        self.assertIn('state', result)
        self.assertIn('predictions', result)
        self.assertIn('errors', result)
        self.assertIn('reward', result)
        self.assertIn('task', result)
        self.assertIn('execution', result)
        self.assertIn('safety', result)
        self.assertIn('duration_seconds', result)
        agent.stop()
        os.chdir(orig_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_error_evaluator_consistency(self):
        """误差评估器输入输出应数学一致。"""
        evaluator = ErrorEvaluator(metrics=['cpu', 'mem'])
        predictions = {'cpu': 50.0, 'mem': 60.0}
        actuals = {'cpu': 55.0, 'mem': 58.0}
        result = evaluator.evaluate(predictions, actuals)
        self.assertEqual(result['errors']['cpu'], 5.0)
        self.assertEqual(result['errors']['mem'], 2.0)
        self.assertEqual(result['mae'], 3.5)
        self.assertEqual(result['mse'], 14.5)
        self.assertEqual(result['max_error'], 5.0)

    def test_prediction_error_reward_chain(self):
        """预测→误差→奖励链条应数学一致。"""
        low = LowLevelPredictor(metrics=['cpu'])
        evaluator = ErrorEvaluator(metrics=['cpu'])

        low.update({'cpu': 50.0})
        predictions = low.predict()
        actuals = {'cpu': 55.0}
        error_result = evaluator.evaluate(predictions, actuals)

        self.assertEqual(error_result['errors']['cpu'], abs(predictions['cpu'] - actuals['cpu']))
        self.assertGreaterEqual(error_result['mae'], 0.0)


if __name__ == '__main__':
    unittest.main()
