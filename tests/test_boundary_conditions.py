#!/usr/bin/env python3
"""
阶段70：极端边界条件与异常输入测试
目标：验证系统对空值、非法值、极大极小值、类型错误等边界条件的处理能力
"""

import os
import sys
import unittest
import tempfile
import shutil
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.prediction.base_predictor import AdaptiveExponentialSmoother, SimpleMovingAverage
from src.prediction.error_evaluator import ErrorEvaluator
from src.drive.reward_engine import RewardEngine
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType
from src.safety.guardian import ResourceGuardian, BehaviorGuardian
from src.optimization.micro_optimizer import MicroOptimizer


class TestNoneAndEmptyInputs(unittest.TestCase):
    """空值与None输入测试"""

    def test_error_evaluator_none_predictions(self):
        """ErrorEvaluator接受None预测不应崩溃"""
        ev = ErrorEvaluator()
        result = ev.evaluate(None, {'cpu_percent': 10.0})
        self.assertIn('mae', result)

    def test_error_evaluator_empty_dicts(self):
        """ErrorEvaluator空字典不应崩溃"""
        ev = ErrorEvaluator()
        result = ev.evaluate({}, {})
        self.assertEqual(result['mae'], 0.0)

    def test_reward_engine_none_state(self):
        """RewardEngine接受None状态不应崩溃"""
        re = RewardEngine()
        result = re.compute_total_reward(None, [0.1, 0.2])
        self.assertIn('total_reward', result)

    def test_reward_engine_empty_errors(self):
        """RewardEngine空误差列表不应崩溃"""
        re = RewardEngine()
        result = re.compute_total_reward({'cpu_percent': 10.0}, [])
        self.assertIn('total_reward', result)

    def test_smoother_update_none(self):
        """平滑器更新None值应忽略或处理"""
        s = AdaptiveExponentialSmoother()
        # 不应崩溃
        s.update(None)  # type: ignore
        preds = s.predict(steps=1)
        self.assertEqual(len(preds), 1)

    def test_task_pool_update_none(self):
        """TaskPool更新None任务列表不应崩溃"""
        pool = TaskPool()
        pool.update(None)  # type: ignore
        stats = pool.get_stats()
        self.assertGreaterEqual(stats['total_tasks'], 0)

    def test_behavior_guardian_none_path(self):
        """BehaviorGuardian检查None路径不应崩溃"""
        bg = BehaviorGuardian()
        result = bg.check_path(None)  # type: ignore
        self.assertTrue(result)

    def test_short_term_memory_store_none(self):
        """ShortTermMemory存储None应优雅处理"""
        mem = ShortTermMemory(db_path=':memory:')
        # store_sample期望字典，传入None应不崩溃
        try:
            mem.store_sample(None)  # type: ignore
        except (TypeError, AttributeError):
            pass  # 允许抛出类型错误，但不应崩溃进程
        mem.close()


class TestExtremeNumericValues(unittest.TestCase):
    """极端数值测试"""

    def test_smoother_update_inf(self):
        """平滑器更新Inf值"""
        s = AdaptiveExponentialSmoother()
        s.update(float('inf'))
        preds = s.predict(steps=1)
        self.assertTrue(math.isfinite(preds[0]) or math.isinf(preds[0]))

    def test_smoother_update_nan(self):
        """平滑器更新NaN值"""
        s = AdaptiveExponentialSmoother()
        s.update(float('nan'))
        preds = s.predict(steps=1)
        # 预测不应为NaN（或至少不应崩溃）
        self.assertTrue(isinstance(preds[0], float))

    def test_error_evaluator_extreme_values(self):
        """误差评估极大极小值"""
        ev = ErrorEvaluator()
        result = ev.evaluate(
            {'cpu': 1e308, 'mem': -1e308},
            {'cpu': -1e308, 'mem': 1e308}
        )
        self.assertIn('mae', result)
        self.assertTrue(math.isfinite(result['mae']) or math.isinf(result['mae']))

    def test_reward_engine_extreme_cpu(self):
        """RewardEngine极端CPU值"""
        re = RewardEngine()
        result = re.compute_total_reward(
            {'cpu_percent': 1e6, 'mem_percent': -1e6},
            [0.1]
        )
        self.assertIn('total_reward', result)
        self.assertTrue(math.isfinite(result['total_reward']))

    def test_resource_guardian_extreme_limits(self):
        """ResourceGuardian极端资源限制"""
        rg = ResourceGuardian(max_cpu_percent=1e-9, max_mem_percent=1e-9)
        result = rg.check_resources()
        self.assertIn('is_safe', result)
        # 几乎不可能满足的限制，应报告不安全
        self.assertFalse(result['is_safe'])

    def test_micro_optimizer_extreme_errors(self):
        """MicroOptimizer极端误差值"""
        opt = MicroOptimizer()
        # 不应崩溃
        opt.record_errors({'cpu': 1e308, 'mem': float('nan')})
        opt.optimize(None, {'cpu': float('inf'), 'mem': -float('inf')})


class TestInvalidTypesAndMalformedData(unittest.TestCase):
    """非法类型与畸形数据测试"""

    def test_error_evaluator_string_values(self):
        """ErrorEvaluator字符串数值"""
        ev = ErrorEvaluator()
        try:
            result = ev.evaluate({'cpu': 'abc'}, {'cpu': 'def'})
        except (TypeError, ValueError):
            pass  # 允许抛出异常

    def test_task_creation_invalid_id(self):
        """Task创建非法ID类型"""
        try:
            t = Task(id=None, type=TaskType.EXPLORE, name='test')  # type: ignore
            self.assertIsNotNone(t.id)
        except (TypeError, AttributeError):
            pass

    def test_config_set_invalid_nested_path(self):
        """Config设置非法嵌套路径"""
        Config._instance = None
        Config._config = {}
        cfg = Config()
        # 不应崩溃
        cfg.set('a.b.c.d.e', 'value')
        self.assertEqual(cfg.get('a.b.c.d.e'), 'value')

    def test_short_term_memory_store_missing_fields(self):
        """ShortTermMemory存储缺失字段的样本"""
        mem = ShortTermMemory(db_path=':memory:', batch_size=1, flush_interval=0.0)
        mem.store_sample({})  # 空字典，所有字段缺失
        count = mem.get_sample_count()
        mem.close()
        self.assertEqual(count, 1)

    def test_short_term_memory_store_extra_fields(self):
        """ShortTermMemory存储多余字段的样本"""
        mem = ShortTermMemory(db_path=':memory:', batch_size=1, flush_interval=0.0)
        mem.store_sample({'unknown_field': 'value', 'reward': 1.0})
        count = mem.get_sample_count()
        mem.close()
        self.assertEqual(count, 1)


class TestAgentLoopBoundaryScenarios(unittest.TestCase):
    """AgentLoop边界场景测试"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        self.temp_dir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)
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
memory:
  max_short_term_samples: 10
logging:
  level: WARNING
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
dashboard:
  enabled: false
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_with_zero_exploration(self):
        """探索率为0时系统仍应运行"""
        agent = AgentLoop('config.yaml')
        agent.task_generator.exploration_ratio = 0.0
        for _ in range(20):
            agent.run_cycle()
        agent.stop()
        self.assertTrue(True)

    def test_agent_with_100_percent_exploration(self):
        """探索率为1时系统仍应运行"""
        agent = AgentLoop('config.yaml')
        agent.task_generator.exploration_ratio = 1.0
        for _ in range(20):
            agent.run_cycle()
        agent.stop()
        self.assertTrue(True)

    def test_agent_with_empty_task_pool(self):
        """空任务池时系统仍应运行"""
        agent = AgentLoop('config.yaml')
        # 清空任务池
        for tid in list(agent.task_pool.tasks.keys()):
            agent.task_pool.remove_task(tid)
        for _ in range(20):
            agent.run_cycle()
        agent.stop()
        self.assertTrue(True)

    def test_agent_run_cycle_after_stop(self):
        """stop后再调用run_cycle不应崩溃"""
        agent = AgentLoop('config.yaml')
        agent.run_cycle()
        agent.stop()
        # stop后再运行，可能静默失败或正常执行
        try:
            agent.run_cycle()
        except Exception:
            pass
        self.assertTrue(True)

    def test_agent_multiple_stops(self):
        """多次调用stop不应崩溃"""
        agent = AgentLoop('config.yaml')
        agent.stop()
        agent.stop()
        agent.stop()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
