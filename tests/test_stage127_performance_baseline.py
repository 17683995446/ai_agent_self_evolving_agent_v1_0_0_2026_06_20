"""阶段127：性能基准与回归测试自动化

建立关键操作的性能基线，防止性能回退。
基线指标包括：
- run_cycle 单次延迟
- 预测器 predict/update 延迟
- ShortTermMemory store/flush 吞吐
- SnapshotManager create/load/delete 延迟
- TaskPool 操作延迟
"""
import os
import sys
import time
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.prediction.base_predictor import ExponentialSmoother, SimpleMovingAverage, AdaptiveExponentialSmoother
from src.prediction.arima_predictor import ARIMAModel
from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.task.task_pool import TaskPool
from src.task.task_generator import TaskGenerator
from src.task.task_definition import Task
from src.core.config import Config


class _PerfTestBase(unittest.TestCase):
    """性能测试基类。"""

    def setUp(self):
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _measure_latency(self, func, iterations=10):
        """测量函数平均延迟（毫秒）。"""
        durations = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            durations.append((time.perf_counter() - start) * 1000)
        return sum(durations) / len(durations)

    def _make_agent(self):
        config = {
            'base_weights': {'safe': 0.4, 'gain': 0.4, 'eff': 0.2},
            'resource_limits': {
                'max_cpu_percent': 80.0,
                'max_mem_percent': 80.0,
                'max_db_size_mb': 500,
                'max_snapshot_count': 10,
            },
            'safety': {
                'max_consecutive_failures': 5,
                'forbidden_paths': [],
                'max_model_restructure_iterations': 3,
            },
            'review': {
                'interval_hours': 6,
                'min_samples': 100,
                'idle_cpu_threshold': 10.0,
                'idle_duration_seconds': 300,
            },
            'memory': {
                'max_short_term_samples': 1000,
                'value_retention_ratio': 0.3,
                'history_window': 100,
            },
            'logging': {
                'level': 'WARNING',
                'file': 'agent.log',
                'max_bytes': 10485760,
                'backup_count': 5,
            },
            'paths': {
                'data_dir': self.temp_dir,
                'snapshot_dir': os.path.join(self.temp_dir, 'snapshots'),
                'log_dir': os.path.join(self.temp_dir, 'logs'),
            },
            'auto_rollback': {'mae_degradation_threshold': 0.2},
            'task_pool': {'max_size': 50},
        }
        import yaml
        config_path = os.path.join(self.temp_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(yaml.dump(config, default_flow_style=False))
        return AgentLoop(config_path)

    def _cleanup_agent(self, agent):
        try:
            if hasattr(agent, 'memory') and agent.memory:
                agent.memory.close()
        except Exception:
            pass
        try:
            if hasattr(agent, '_remove_log_handler'):
                agent._remove_log_handler()
        except Exception:
            pass
        try:
            agent.stop()
        except Exception:
            pass


class TestAgentLoopCycleLatency(_PerfTestBase):
    """验证 AgentLoop run_cycle 延迟基线。"""

    def test_run_cycle_latency_under_baseline(self):
        """单次 run_cycle 延迟应低于 500ms。"""
        agent = self._make_agent()
        try:
            # warmup
            agent.run_cycle()
            latency = self._measure_latency(agent.run_cycle, iterations=5)
            self.assertLess(latency, 500,
                            f"run_cycle 平均延迟 {latency:.2f}ms 超过 500ms 基线")
        finally:
            self._cleanup_agent(agent)


class TestPredictorLatency(_PerfTestBase):
    """验证预测器操作延迟基线。"""

    def test_exponential_smoother_predict_latency(self):
        """ExponentialSmoother predict 延迟应低于 1ms。"""
        predictor = ExponentialSmoother(alpha=0.3)
        for i in range(50):
            predictor.update(float(i))
        latency = self._measure_latency(lambda: predictor.predict(), iterations=100)
        self.assertLess(latency, 1.0,
                        f"ExponentialSmoother.predict 平均延迟 {latency:.4f}ms 超过 1ms 基线")

    def test_simple_moving_average_predict_latency(self):
        """SimpleMovingAverage predict 延迟应低于 1ms。"""
        predictor = SimpleMovingAverage(window=20)
        for i in range(50):
            predictor.update(float(i))
        latency = self._measure_latency(lambda: predictor.predict(), iterations=100)
        self.assertLess(latency, 1.0,
                        f"SimpleMovingAverage.predict 平均延迟 {latency:.4f}ms 超过 1ms 基线")

    def test_arima_predict_latency_under_baseline(self):
        """ARIMAModel predict 延迟应低于 100ms。"""
        predictor = ARIMAModel(order=(1, 1, 1), fit_interval=10)
        for i in range(50):
            predictor.update(float(i))
        latency = self._measure_latency(lambda: predictor.predict(), iterations=10)
        self.assertLess(latency, 100.0,
                        f"ARIMAModel.predict 平均延迟 {latency:.2f}ms 超过 100ms 基线")


class TestShortTermMemoryThroughput(_PerfTestBase):
    """验证 ShortTermMemory 吞吐基线。"""

    def test_store_sample_throughput(self):
        """store_sample 吞吐应高于 1000 ops/sec。"""
        db_path = os.path.join(self.temp_dir, 'perf.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=10000, batch_size=100)
        try:
            start = time.perf_counter()
            iterations = 500
            for i in range(iterations):
                mem.store_sample({'cycle': i, 'value': float(i), 'timestamp': time.time()})
            duration = time.perf_counter() - start
            throughput = iterations / duration
            self.assertGreater(throughput, 1000,
                               f"store_sample 吞吐 {throughput:.2f} ops/sec 低于 1000 ops/sec 基线")
        finally:
            mem.close()

    def test_flush_latency_under_baseline(self):
        """批量 flush 延迟应低于 50ms。"""
        db_path = os.path.join(self.temp_dir, 'perf.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=10000, batch_size=100)
        try:
            for i in range(100):
                mem.store_sample({'cycle': i, 'value': float(i), 'timestamp': time.time()})
            latency = self._measure_latency(mem._flush_all, iterations=5)
            self.assertLess(latency, 50.0,
                            f"_flush_all 平均延迟 {latency:.2f}ms 超过 50ms 基线")
        finally:
            mem.close()


class TestSnapshotManagerLatency(_PerfTestBase):
    """验证 SnapshotManager 操作延迟基线。"""

    def test_create_snapshot_latency(self):
        """create_snapshot 延迟应低于 50ms。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        state = {'weights': [0.1] * 100}
        latency = self._measure_latency(lambda: sm.create_snapshot(state, tag='perf'), iterations=10)
        self.assertLess(latency, 50.0,
                        f"create_snapshot 平均延迟 {latency:.2f}ms 超过 50ms 基线")

    def test_load_snapshot_latency(self):
        """load_snapshot 延迟应低于 50ms。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        state = {'weights': [0.1] * 100}
        sid = sm.create_snapshot(state, tag='perf')
        latency = self._measure_latency(lambda: sm.load_snapshot(sid), iterations=10)
        self.assertLess(latency, 50.0,
                        f"load_snapshot 平均延迟 {latency:.2f}ms 超过 50ms 基线")


class TestTaskPoolLatency(_PerfTestBase):
    """验证 TaskPool 操作延迟基线。"""

    def test_task_pool_add_select_latency(self):
        """task_pool add + select 延迟应低于 10ms。"""
        pool = TaskPool(max_size=50)
        generator = TaskGenerator()
        tasks = generator.generate({'cpu': 0.5, 'mem': 0.3, 'disk': 0.2})
        pool.update(tasks)
        latency = self._measure_latency(pool.select_task, iterations=100)
        self.assertLess(latency, 10.0,
                        f"select_task 平均延迟 {latency:.4f}ms 超过 10ms 基线")


class TestPerformanceRegression(_PerfTestBase):
    """验证性能不随周期数显著衰减。"""

    def test_cycle_duration_regression(self):
        """连续 20 个 cycle 的延迟不应持续增长。"""
        agent = self._make_agent()
        try:
            durations = []
            for _ in range(20):
                start = time.perf_counter()
                agent.run_cycle()
                durations.append((time.perf_counter() - start) * 1000)

            first_half = sum(durations[:10]) / 10
            second_half = sum(durations[10:]) / 10
            regression_ratio = second_half / first_half if first_half > 0 else 1.0
            self.assertLess(regression_ratio, 3.0,
                            f"cycle 延迟后/前比值 {regression_ratio:.2f} 超过 3x，存在性能回归")
        finally:
            self._cleanup_agent(agent)


if __name__ == '__main__':
    unittest.main()
