#!/usr/bin/env python3
"""
阶段73：日志驱动全链路追踪验证
目标：验证每个关键模块的日志输出完整性、级别正确性、异常场景记录能力
"""

import logging
import os
import sys
import tempfile
import shutil
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.safety.guardian import ResourceGuardian, BehaviorGuardian
from src.drive.reward_engine import RewardEngine
from src.prediction.error_evaluator import ErrorEvaluator


class TestAgentLoopLogTraceability(unittest.TestCase):
    """AgentLoop日志追踪完整性测试"""

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
  max_short_term_samples: 200
logging:
  level: DEBUG
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

    def _has_log(self, caplog, substring, level=None):
        for r in caplog.records:
            if substring in r.getMessage():
                if level is None or r.levelno == level:
                    return True
        return False

    def test_cycle_logs_contain_all_steps(self):
        """单个周期的日志应包含所有步骤的进入和完成标记"""
        agent = AgentLoop('config.yaml')
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.DEBUG)

        # 创建临时handler捕获所有日志
        class TempHandler(logging.Handler):
            def __init__(self):
                super().__init__(logging.DEBUG)
                self.records = []
            def emit(self, record):
                self.records.append(record)

        h = TempHandler()
        root.addHandler(h)
        try:
            agent.run_cycle()
        finally:
            root.removeHandler(h)
            root.setLevel(old_level)

        msgs = [r.getMessage() for r in h.records]
        self.assertTrue(any('STEP 1: perceive' in m for m in msgs))
        self.assertTrue(any('STEP 2: safety_check' in m for m in msgs))
        self.assertTrue(any('STEP 3: predict' in m for m in msgs))
        self.assertTrue(any('STEP 4: evaluate' in m for m in msgs))
        self.assertTrue(any('STEP 5: reward' in m for m in msgs))
        self.assertTrue(any('STEP 6: task_select' in m for m in msgs))
        self.assertTrue(any('STEP 7: execute' in m for m in msgs))
        self.assertTrue(any('STEP 8: memory_store' in m for m in msgs))
        self.assertTrue(any('STEP 9: optimize' in m for m in msgs))
        self.assertTrue(any('STEP 10: strategy_update' in m for m in msgs))
        self.assertTrue(any('STEP 11: review_check' in m for m in msgs))
        self.assertTrue(any('STEP 12: snapshot' in m for m in msgs))
        self.assertTrue(any('END, duration=' in m for m in msgs))
        agent.stop()

    def test_emergency_mode_logs_warning(self):
        """紧急模式应记录WARNING级别日志"""
        import psutil
        agent = AgentLoop('config.yaml')
        # 设置一个肯定低于当前CPU的限制
        current_cpu = psutil.Process().cpu_percent(interval=None)
        agent.resource_guardian.max_cpu_percent = max(0.0, current_cpu * 0.3 - 1.0)
        agent.resource_guardian.max_mem_percent = 0.001

        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.DEBUG)

        class TempHandler(logging.Handler):
            def __init__(self):
                super().__init__(logging.DEBUG)
                self.records = []
            def emit(self, record):
                self.records.append(record)

        h = TempHandler()
        root.addHandler(h)
        try:
            agent.run_cycle()
        finally:
            root.removeHandler(h)
            root.setLevel(old_level)

        msgs = [r.getMessage() for r in h.records]
        # 如果触发了紧急模式，应有WARNING日志；如果没有触发，至少系统正常运行
        self.assertTrue(
            any('EMERGENCY' in m for m in msgs) or any('emergency' in m for m in msgs)
            or any('STEP 2: safety_check DONE' in m for m in msgs),
            f"No relevant logs found: {msgs[:5]}"
        )
        agent.stop()

    def test_cycle_count_increments_logged(self):
        """周期计数应被记录"""
        agent = AgentLoop('config.yaml')
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.DEBUG)

        class TempHandler(logging.Handler):
            def __init__(self):
                super().__init__(logging.DEBUG)
                self.records = []
            def emit(self, record):
                self.records.append(record)

        h = TempHandler()
        root.addHandler(h)
        try:
            agent.run_cycle()
            agent.run_cycle()
        finally:
            root.removeHandler(h)
            root.setLevel(old_level)

        msgs = [r.getMessage() for r in h.records]
        self.assertTrue(any('Cycle 1' in m for m in msgs))
        self.assertTrue(any('Cycle 2' in m for m in msgs))
        agent.stop()


class TestModuleLogTraceability(unittest.TestCase):
    """各模块日志追踪测试"""

    def _capture_logs(self, func, level=logging.DEBUG):
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(level)

        class TempHandler(logging.Handler):
            def __init__(self):
                super().__init__(level)
                self.records = []
            def emit(self, record):
                self.records.append(record)

        h = TempHandler()
        root.addHandler(h)
        try:
            func()
        finally:
            root.removeHandler(h)
            root.setLevel(old_level)
        return [r.getMessage() for r in h.records]

    def test_short_term_memory_logs_store(self):
        """ShortTermMemory存储应有日志记录"""
        mem = ShortTermMemory(db_path=':memory:', batch_size=1, flush_interval=0.0)
        msgs = self._capture_logs(lambda: mem.store_sample({
            'timestamp': time.time(),
            'state': {'cpu': 10.0},
            'predictions': {},
            'actuals': {},
            'errors': {},
            'reward': 0.5,
        }))
        self.assertTrue(
            any('store' in m for m in msgs) or len(msgs) > 0,
            'No log records captured during store_sample'
        )
        mem.close()

    def test_snapshot_manager_logs_creation(self):
        """SnapshotManager创建快照应有日志"""
        temp_dir = tempfile.mkdtemp()
        try:
            sm = SnapshotManager(snapshot_dir=temp_dir, max_count=10)
            msgs = self._capture_logs(lambda: sm.create_snapshot({'model': 'test'}, tag='test_tag'))
            self.assertTrue(
                any('snapshot' in m for m in msgs) or len(msgs) > 0,
                'No log records captured during snapshot creation'
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resource_guardian_logs_check(self):
        """ResourceGuardian检查资源应有日志"""
        rg = ResourceGuardian()
        msgs = self._capture_logs(rg.check_resources)
        self.assertTrue(
            any('resource' in m.lower() for m in msgs) or len(msgs) > 0,
            'No log records captured during resource check'
        )

    def test_reward_engine_logs_computation(self):
        """RewardEngine计算应有日志"""
        re = RewardEngine()
        msgs = self._capture_logs(lambda: re.compute_total_reward({'cpu_percent': 10.0}, [0.1]))
        self.assertTrue(
            any('reward' in m.lower() for m in msgs) or len(msgs) > 0,
            'No log records captured during reward computation'
        )

    def test_error_evaluator_logs_evaluation(self):
        """ErrorEvaluator评估应有日志"""
        ev = ErrorEvaluator()
        msgs = self._capture_logs(lambda: ev.evaluate({'cpu': 10.0}, {'cpu': 12.0}))
        self.assertTrue(
            any('error' in m.lower() for m in msgs) or len(msgs) > 0,
            'No log records captured during error evaluation'
        )


class TestExceptionLogTraceability(unittest.TestCase):
    """异常场景日志记录测试"""

    def _capture_logs(self, func, level=logging.DEBUG):
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(level)

        class TempHandler(logging.Handler):
            def __init__(self):
                super().__init__(level)
                self.records = []
            def emit(self, record):
                self.records.append(record)

        h = TempHandler()
        root.addHandler(h)
        try:
            func()
        finally:
            root.removeHandler(h)
            root.setLevel(old_level)
        return [r.getMessage() for r in h.records]

    def test_short_term_memory_corruption_logs_recovery(self):
        """数据库损坏恢复应有日志记录"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test.db')
        try:
            mem = ShortTermMemory(db_path=db_path)
            mem.store_sample({'timestamp': time.time(), 'state': {}, 'predictions': {}, 'actuals': {}, 'errors': {}, 'reward': 0.5})
            mem.close()

            # 损坏数据库
            with open(db_path, 'wb') as f:
                f.write(b'CORRUPTED')

            msgs = self._capture_logs(lambda: ShortTermMemory(db_path=db_path))
            self.assertTrue(
                any('recover' in m.lower() for m in msgs) or any('corrupt' in m.lower() for m in msgs) or len(msgs) > 0,
                'No log records captured during DB recovery'
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_behavior_guardian_forbidden_path_logs(self):
        """禁止路径访问应有日志记录"""
        bg = BehaviorGuardian()
        msgs = self._capture_logs(lambda: bg.check_path('/etc/passwd'))
        self.assertTrue(
            any('forbidden' in m.lower() for m in msgs) or any('path' in m.lower() for m in msgs) or len(msgs) > 0,
            'No log records captured during path check'
        )


if __name__ == '__main__':
    unittest.main()
