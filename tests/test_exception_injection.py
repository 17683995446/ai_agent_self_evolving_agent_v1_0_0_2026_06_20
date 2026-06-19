#!/usr/bin/env python3
"""
异常注入测试
目标：在关键路径注入异常，验证系统是否平稳兜底，无崩溃报错
"""

import os
import sys
import unittest
import tempfile
import shutil
import sqlite3
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.safety.guardian import ResourceGuardian, BehaviorGuardian


class TestExceptionInjection(unittest.TestCase):
    """异常注入测试"""

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
  interval_hours: 0.001
  min_samples: 5
memory:
  max_short_term_samples: 200
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

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predictor_exception_does_not_crash_cycle(self):
        """预测器抛出异常不应导致周期崩溃"""
        agent = AgentLoop('config.yaml')

        with patch.object(agent.low_predictor, 'predict', side_effect=RuntimeError('predictor crash')):
            try:
                result = agent.run_cycle()
                self.assertIn('cycle', result)
            except RuntimeError:
                self.fail("Predictor exception should be caught, not crash the cycle")

        agent.stop()

    def test_memory_exception_does_not_crash_cycle(self):
        """内存存储抛出异常不应导致周期崩溃"""
        agent = AgentLoop('config.yaml')

        with patch.object(agent.memory, 'store_sample', side_effect=sqlite3.Error('db locked')):
            try:
                result = agent.run_cycle()
                self.assertIn('cycle', result)
            except Exception:
                self.fail("Memory exception should be caught")

        agent.stop()

    def test_reward_engine_exception_does_not_crash_cycle(self):
        """驱力计算抛出异常不应导致周期崩溃"""
        agent = AgentLoop('config.yaml')

        with patch.object(agent.reward_engine, 'compute_total_reward', side_effect=ValueError('invalid state')):
            try:
                result = agent.run_cycle()
                self.assertIn('cycle', result)
            except ValueError:
                self.fail("Reward engine exception should be caught")

        agent.stop()

    def test_task_execution_exception_does_not_crash_cycle(self):
        """任务执行抛出异常不应导致周期崩溃"""
        agent = AgentLoop('config.yaml')

        with patch.object(agent, '_execute_task', side_effect=Exception('task crash')):
            try:
                result = agent.run_cycle()
                self.assertIn('cycle', result)
            except Exception:
                self.fail("Task execution exception should be caught")

        agent.stop()

    def test_safety_check_exception_does_not_crash_cycle(self):
        """安全检查抛出异常不应导致周期崩溃"""
        agent = AgentLoop('config.yaml')

        with patch.object(agent.safety_enforcer, 'full_check', side_effect=OSError('resource check failed')):
            try:
                result = agent.run_cycle()
                self.assertIn('cycle', result)
            except OSError:
                self.fail("Safety check exception should be caught")

        agent.stop()

    def test_snapshot_exception_does_not_crash_cycle(self):
        """快照创建抛出异常不应导致周期崩溃"""
        agent = AgentLoop('config.yaml')

        with patch.object(agent.snapshot_manager, 'create_snapshot', side_effect=IOError('disk full')):
            try:
                result = agent.run_cycle()
                self.assertIn('cycle', result)
            except IOError:
                self.fail("Snapshot exception should be caught")

        agent.stop()

    def test_sensor_exception_does_not_crash_perceive(self):
        """传感器异常不应导致感知崩溃"""
        agent = AgentLoop('config.yaml')

        with patch.object(agent.state_vector.system_sensor, 'perceive', side_effect=PermissionError('sensor denied')):
            try:
                result = agent.run_cycle()
                self.assertIn('cycle', result)
            except PermissionError:
                self.fail("Sensor exception should be caught")

        agent.stop()

    def test_consecutive_exceptions_agent_survives(self):
        """连续异常注入，Agent仍能存活"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(20):
            try:
                # 每5个周期注入不同异常
                if i % 5 == 0:
                    with patch.object(agent.low_predictor, 'predict', side_effect=RuntimeError('predictor error')):
                        agent.run_cycle()
                elif i % 5 == 1:
                    with patch.object(agent.memory, 'store_sample', side_effect=Exception('memory error')):
                        agent.run_cycle()
                elif i % 5 == 2:
                    with patch.object(agent.reward_engine, 'compute_total_reward', side_effect=ValueError('reward error')):
                        agent.run_cycle()
                else:
                    agent.run_cycle()
            except Exception as e:
                errors.append(f"cycle_{i}: {e}")

        agent.stop()
        self.assertEqual(len(errors), 0, f"Consecutive exceptions crashed agent: {errors}")

    def test_resource_guardian_with_none_state(self):
        """资源护栏接收None状态不应崩溃"""
        guardian = ResourceGuardian(data_dir=self.temp_dir)

        try:
            result = guardian.check_resources()
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"ResourceGuardian crashed on None state: {e}")

    def test_behavior_guardian_with_none_path(self):
        """行为护栏接收None路径不应崩溃"""
        guardian = BehaviorGuardian()

        try:
            result = guardian.check_path(None)
            # None路径可能被允许也可能被拒绝，关键是不要崩溃
            self.assertTrue(result is True or result is False)
        except Exception as e:
            self.fail(f"BehaviorGuardian crashed on None path: {e}")

    def test_short_term_memory_with_corrupt_data(self):
        """短期记忆接收损坏数据不应崩溃"""
        db_path = os.path.join(self.temp_dir, 'data', 'corrupt.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=100)

        corrupt_samples = [
            None,
            {},
            {'reward': float('nan')},
            {'reward': float('inf')},
            {'reward': 'not_a_number'},
            {'reward': None},
        ]

        errors = []
        for sample in corrupt_samples:
            try:
                mem.store_sample(sample)
            except Exception as e:
                errors.append(str(e))

        mem.close()
        self.assertEqual(len(errors), 0, f"Corrupt data crashed memory: {errors}")


if __name__ == '__main__':
    unittest.main()
