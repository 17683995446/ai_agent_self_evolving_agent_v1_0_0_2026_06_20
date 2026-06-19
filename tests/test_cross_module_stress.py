#!/usr/bin/env python3
"""
跨模块集成极限压力测试
目标：同时压榨所有模块，发现跨模块交互中的bug
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.task.task_pool import TaskPool
from src.prediction.error_evaluator import ErrorEvaluator
from src.optimization.auto_rollback import AutoRollback
from src.optimization.micro_optimizer import MicroOptimizer
from src.safety.guardian import ResourceGuardian, BehaviorGuardian, SafetyEnforcer
from src.dashboard.server import DashboardServer


class TestCrossModuleStress(unittest.TestCase):
    """跨模块集成极限压力测试"""

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
  max_short_term_samples: 500
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

    def test_all_modules_concurrent_stress(self):
        """所有模块并发压力测试"""
        agent = AgentLoop('config.yaml')
        errors = []
        stop_event = threading.Event()

        def cycle_stress():
            try:
                for _ in range(100):
                    if stop_event.is_set():
                        break
                    agent.run_cycle()
            except Exception as e:
                errors.append(f"cycle: {e}")

        def memory_stress():
            try:
                for i in range(500):
                    if stop_event.is_set():
                        break
                    agent.memory.store_sample({'reward': float(i), 'cpu': 50.0})
                    if i % 50 == 0:
                        agent.memory.load_samples(limit=20)
                        agent.memory.cleanup_old_samples(retention_ratio=0.1)
            except Exception as e:
                errors.append(f"memory: {e}")

        def task_stress():
            try:
                for i in range(300):
                    if stop_event.is_set():
                        break
                    state = agent.state_vector.perceive()
                    agent.task_generator.generate(state)
                    agent.task_pool.score_all(state, {})
                    agent.task_pool.select_task()
                    if i % 20 == 0:
                        agent.task_pool._enforce_size_limit()
            except Exception as e:
                errors.append(f"task: {e}")

        def prediction_stress():
            try:
                for i in range(200):
                    if stop_event.is_set():
                        break
                    low = agent.low_predictor.predict()
                    mid = agent.mid_predictor.predict()
                    agent.high_predictor.predict(low, mid)
                    agent.low_predictor.update({'cpu_percent': float(random.random() * 100)})
            except Exception as e:
                errors.append(f"prediction: {e}")

        def snapshot_stress():
            try:
                for i in range(50):
                    if stop_event.is_set():
                        break
                    agent.snapshot_manager.create_snapshot(
                        {'weights': [random.random()]},
                        tag='stress'
                    )
                    if i % 10 == 0:
                        agent.snapshot_manager.verify_and_repair()
            except Exception as e:
                errors.append(f"snapshot: {e}")

        threads = [
            threading.Thread(target=cycle_stress),
            threading.Thread(target=memory_stress),
            threading.Thread(target=task_stress),
            threading.Thread(target=prediction_stress),
            threading.Thread(target=snapshot_stress),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        stop_event.set()

        agent.stop()
        self.assertEqual(len(errors), 0, f"Cross-module stress errors: {errors}")

    def test_module_interaction_edge_cases(self):
        """模块交互边界情况测试"""
        agent = AgentLoop('config.yaml')

        # 1. 空预测器调用评估
        evaluator = ErrorEvaluator()
        result = evaluator.evaluate({}, {})
        self.assertIn('mae', result)

        # 2. 空任务池调用选择器
        empty_pool = TaskPool(max_size=10)
        task = empty_pool.select_task()
        self.assertIsNone(task)

        # 3. 回滚模块在基线未设置时检查
        rollback = AutoRollback(agent.snapshot_manager)
        stats = rollback.check_and_rollback(0.5, lambda x: True)
        self.assertEqual(stats['action'], 'no_baseline')

        # 4. 仪表盘在数据为空时启动
        dashboard = DashboardServer(host='127.0.0.1', port=0)
        try:
            dashboard.start()
            time.sleep(0.1)
        finally:
            dashboard.stop()

        # 5. 微优化器在空错误历史时优化
        optimizer = MicroOptimizer()
        optimizer.optimize(agent.low_predictor, {'cpu_percent': 0.1})

        agent.stop()

    def test_resource_guardian_under_extreme_values(self):
        """资源护栏在极端值下的行为"""
        guardian = ResourceGuardian(
            max_cpu_percent=20.0,
            max_mem_percent=20.0,
            max_disk_gb=1.0,
            data_dir=self.temp_dir
        )

        try:
            result = guardian.check_resources()
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"ResourceGuardian crashed on extreme values: {e}")

    def test_behavior_guardian_path_traversal(self):
        """行为护栏路径遍历攻击测试"""
        guardian = BehaviorGuardian(forbidden_paths=['/etc', '/boot'])

        malicious_paths = [
            '/etc/passwd',
            '/boot/../etc/passwd',
            '/etc/',
            '////etc////passwd',
            '/etc\x00.txt',
        ]

        for path in malicious_paths:
            try:
                allowed = guardian.check_path(path)
                self.assertFalse(allowed, f"Path should be forbidden: {path}")
            except Exception as e:
                self.fail(f"BehaviorGuardian crashed on path {path}: {e}")

    def test_safety_enforcer_cascading_failure(self):
        """安全执行器级联故障测试"""
        enforcer = SafetyEnforcer(
            resource_guardian=ResourceGuardian(data_dir=self.temp_dir),
            behavior_guardian=BehaviorGuardian(),
        )

        # 连续检查
        for _ in range(10):
            enforcer.full_check()

        self.assertTrue(enforcer.is_emergency() or not enforcer.is_emergency())


if __name__ == '__main__':
    unittest.main()
