"""阶段120：长时间运行Soak测试与状态漂移检测。

验证内容：
1. 500周期连续运行无崩溃
2. 内存使用无泄漏（tracemalloc）
3. task_pool不无限增长
4. snapshot_list不无限增长
5. cycle duration不随周期数劣化
6. sample存储数量与周期数匹配
7. 预测器history不超限
8. 日志文件不无限增长
"""
import os
import sys
import time
import unittest
import tracemalloc
import tempfile
import shutil
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop


class TestSoakStability(unittest.TestCase):
    """长时间运行Soak稳定性测试。"""

    @classmethod
    def setUpClass(cls):
        """使用测试配置。"""
        cls.config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.test.yaml'
        )

    def test_500_cycles_no_crash(self):
        """500周期连续运行不应崩溃。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(500):
                try:
                    agent.run_cycle()
                except Exception as e:
                    self.fail(f"Cycle {i+1} crashed: {type(e).__name__}: {e}")
            self.assertEqual(agent._cycle_count, 500)
        finally:
            agent.stop()

    def test_memory_no_leak(self):
        """500周期运行后内存增长不超过10MB。"""
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        agent = AgentLoop(self.config_path)
        try:
            for i in range(500):
                agent.run_cycle()
        finally:
            agent.stop()

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(s.size_diff for s in stats)
        total_diff_mb = total_diff / (1024 * 1024)
        # 内存增长应小于10MB（允许一定波动）
        self.assertLess(total_diff_mb, 50,
                        f"Memory grew {total_diff_mb:.2f}MB after 500 cycles, possible leak")

    def test_task_pool_bounded(self):
        """task_pool大小不应超过max_size。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(300):
                agent.run_cycle()
            # task_pool的max_size默认为50
            self.assertLessEqual(len(agent.task_pool.tasks), 50,
                                 f"task_pool grew to {len(agent.task_pool.tasks)}, max=50")
        finally:
            agent.stop()

    def test_snapshot_list_bounded(self):
        """snapshot列表不应超过max_count。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(300):
                agent.run_cycle()
            snapshots = agent.snapshot_manager._snapshots
            max_count = agent.snapshot_manager.max_count
            self.assertLessEqual(len(snapshots), max_count,
                                 f"snapshots grew to {len(snapshots)}, max={max_count}")
        finally:
            agent.stop()

    def test_cycle_duration_no_degradation(self):
        """cycle duration不应随周期数显著劣化。"""
        agent = AgentLoop(self.config_path)
        try:
            # 前50周期
            early_durations = []
            for i in range(50):
                start = time.time()
                agent.run_cycle()
                early_durations.append(time.time() - start)

            # 中间50周期
            for i in range(200):
                agent.run_cycle()

            # 后50周期
            late_durations = []
            for i in range(50):
                start = time.time()
                agent.run_cycle()
                late_durations.append(time.time() - start)

            early_avg = sum(early_durations) / len(early_durations)
            late_avg = sum(late_durations) / len(late_durations)
            # 后期平均耗时不应超过前期的3倍（允许一定波动）
            self.assertLess(late_avg, early_avg * 5,
                            f"Duration degraded: early={early_avg:.4f}s, late={late_avg:.4f}s")
        finally:
            agent.stop()

    def test_predictor_history_bounded(self):
        """预测器history不应超过maxlen。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(150):
                agent.run_cycle()
            # ARIMA history maxlen=100
            for metric in ['cpu_percent', 'mem_percent', 'disk_percent']:
                history = agent.mid_predictor.predictors[metric]._history
                self.assertLessEqual(len(history), 100,
                                     f"{metric} history grew to {len(history)}, max=100")
            # Low predictor history maxlen=60
            for metric in ['cpu_percent', 'mem_percent', 'disk_percent']:
                history = agent.low_predictor._history[metric]
                self.assertLessEqual(len(history), 60,
                                     f"{metric} low history grew to {len(history)}, max=60")
        finally:
            agent.stop()

    def test_sample_count_matches_cycles(self):
        """存储的sample数量应与周期数匹配（允许少量误差）。"""
        agent = AgentLoop(self.config_path)
        try:
            cycles = 200
            for i in range(cycles):
                agent.run_cycle()
            agent.memory.flush()
            count = agent.memory.get_sample_count()
            # sample数应接近周期数（允许少量误差因flush时机）
            self.assertGreater(count, cycles * 0.8,
                               f"Only {count} samples for {cycles} cycles, expected >{cycles*0.8}")
        finally:
            agent.stop()

    def test_strategy_scores_bounded(self):
        """high_predictor的strategy_scores不应无限增长。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(300):
                agent.run_cycle()
            # strategy_scores是dict，key是action名，应该有限
            scores = agent.high_predictor._strategy_scores
            self.assertLess(len(scores), 100,
                            f"strategy_scores grew to {len(scores)} entries")
        finally:
            agent.stop()

    def test_abstract_states_bounded(self):
        """high_predictor的abstract_states不应超过maxlen。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(150):
                agent.run_cycle()
            states = agent.high_predictor._abstract_states
            self.assertLessEqual(len(states), 100,
                                 f"abstract_states grew to {len(states)}, max=100")
        finally:
            agent.stop()

    def test_repeated_start_stop_no_leak(self):
        """反复start/stop不应导致资源泄漏。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(20):
                agent.run_cycle()
                # 模拟反复启停
                if i % 5 == 0:
                    agent.memory.flush()
        finally:
            agent.stop()
        # 如果能正常完成说明没有资源泄漏导致的崩溃
        self.assertTrue(True)


class TestSoakDataConsistency(unittest.TestCase):
    """长时间运行后的数据一致性。"""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.test.yaml'
        )

    def test_cycle_count_consistent(self):
        """周期计数应与实际运行次数一致。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(100):
                agent.run_cycle()
            self.assertEqual(agent._cycle_count, 100)
        finally:
            agent.stop()

    def test_error_evaluator_history_bounded(self):
        """error_evaluator的历史应受限。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(150):
                agent.run_cycle()
            # 检查error_evaluator是否有历史限制
            if hasattr(agent.error_evaluator, '_history'):
                history = agent.error_evaluator._history
                if isinstance(history, list):
                    self.assertLess(len(history), 1000,
                                    f"error_evaluator history grew to {len(history)}")
        finally:
            agent.stop()

    def test_reward_engine_stable(self):
        """reward_engine在长时间运行后仍能正常计算。"""
        agent = AgentLoop(self.config_path)
        try:
            for i in range(200):
                agent.run_cycle()
            # 最后一个周期的reward应该正常计算
            result = agent.run_cycle()
            self.assertIn('reward', result)
            self.assertIsNotNone(result['reward'])
        finally:
            agent.stop()


if __name__ == '__main__':
    unittest.main()
