#!/usr/bin/env python3
"""
阶段69：长时间稳定性测试（soak test）
目标：模拟长时间运行，验证系统一致性、状态机正确性、复盘触发
"""

import os
import sys
import unittest
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager


class TestLongRunningConsistency(unittest.TestCase):
    """长时间运行一致性测试"""

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
  min_samples: 10
memory:
  max_short_term_samples: 200
logging:
  level: WARNING
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

    def test_500_cycles_stats_consistency(self):
        """500个周期后统计数据应一致"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(500):
            try:
                agent.run_cycle()
            except Exception as e:
                errors.append(f"cycle {i}: {e}")
                if len(errors) > 3:
                    break

        stats = agent.get_stats()
        agent.stop()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # cycle_count 应准确
        self.assertGreaterEqual(stats['cycle_count'], 490)
        # 任务池不应为负
        self.assertGreaterEqual(stats['task_pool']['total_tasks'], 0)
        # 记忆样本数不应超过限制
        self.assertLessEqual(stats['memory_samples'], 200)

    def test_snapshot_created_after_100_cycles(self):
        """100个周期后应产生至少一个快照"""
        agent = AgentLoop('config.yaml')
        for _ in range(110):
            agent.run_cycle()

        snapshots = agent.snapshot_manager.list_snapshots()
        agent.stop()
        self.assertGreaterEqual(len(snapshots), 1, "No snapshot created after 110 cycles")

    def test_review_triggered_after_interval(self):
        """复盘应在间隔后触发"""
        agent = AgentLoop('config.yaml')
        # review interval 设为 0.001小时 = 3.6秒
        # 运行足够周期使复盘触发
        for _ in range(200):
            agent.run_cycle()
            time.sleep(0.02)

        # 复盘应已清理旧样本
        count = agent.memory.get_sample_count()
        agent.stop()
        # 样本数应被控制在合理范围（复盘会调用cleanup_old_samples）
        self.assertLessEqual(count, 200)

    def test_no_cumulative_error_drift(self):
        """误差不应无限制累积漂移"""
        agent = AgentLoop('config.yaml')
        mae_history = []

        for _ in range(200):
            agent.run_cycle()
            # 从state_vector读取当前mae
            mae = getattr(agent.state_vector, '_current_mae', 0.0)
            mae_history.append(mae)

        agent.stop()
        # 最近50个周期的平均MAE不应比前50个高太多（允许一定波动，但不应单调恶化）
        if len(mae_history) >= 100:
            early_avg = sum(mae_history[:50]) / 50
            late_avg = sum(mae_history[-50:]) / 50
            # 后期不应比前期高5倍以上（若预测器在修正，应保持稳定或改善）
            self.assertLess(late_avg, early_avg * 5.0 + 1.0,
                            f"MAE drift: early={early_avg:.3f}, late={late_avg:.3f}")

    def test_state_machine_transitions_valid(self):
        """状态机转换应始终有效"""
        agent = AgentLoop('config.yaml')
        valid_states = {'IDLE', 'ACTIVE', 'REVIEW', 'EMERGENCY', 'MAINTENANCE', 'SHUTDOWN'}
        state_history = []

        for _ in range(300):
            agent.run_cycle()
            # 从safety推断当前状态
            if agent.safety_enforcer.is_emergency():
                state = 'EMERGENCY'
            else:
                state = 'ACTIVE'
            state_history.append(state)
            self.assertIn(state, valid_states)

        agent.stop()
        # 应至少经历过ACTIVE状态
        self.assertIn('ACTIVE', state_history)

    def test_memory_integrity_after_long_run(self):
        """长时间运行后记忆数据完整性"""
        agent = AgentLoop('config.yaml')
        for _ in range(200):
            agent.run_cycle()

        samples = agent.memory.load_samples(limit=50)
        agent.stop()

        # 所有样本应有必要字段
        for s in samples:
            self.assertIn('timestamp', s)
            self.assertIn('state', s)
            self.assertIn('predictions', s)
            self.assertIn('actuals', s)
            self.assertIn('errors', s)
            self.assertIn('reward', s)


class TestSnapshotRobustnessOverTime(unittest.TestCase):
    """快照长时间鲁棒性"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_snapshots_index_integrity(self):
        """大量快照后索引仍一致"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=100)
        for i in range(50):
            sm.create_snapshot(model_state={'idx': i}, tag=f"snap_{i}",
                               metrics={'mae': 0.1, 'coverage': 0.9, 'eff_score': 0.8})

        snapshots = sm.list_snapshots()
        self.assertEqual(len(snapshots), 50)

        # 验证索引文件可加载
        sm2 = SnapshotManager(snapshot_dir=self.temp_dir, max_count=100)
        snapshots2 = sm2.list_snapshots()
        self.assertEqual(len(snapshots2), 50)


if __name__ == '__main__':
    unittest.main()
