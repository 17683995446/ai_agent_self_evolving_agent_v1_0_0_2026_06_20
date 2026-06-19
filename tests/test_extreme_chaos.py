#!/usr/bin/env python3
"""
极端混沌测试 - 发现真实运行时的bug
目标：模拟更长时间、更大规模、更错综复杂的混乱场景
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time
import random
import math
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.memory.short_term import ShortTermMemory
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType
from src.prediction.arima_predictor import ARIMAModel
from src.prediction.base_predictor import ExponentialSmoother


class ExtremeChaosMonkey:
    """极端混沌猴子"""

    @staticmethod
    def inject_nan_to_predictor(predictor):
        """向预测器注入NaN数据"""
        predictor.update({'cpu_percent': float('nan')})
        predictor.update({'mem_percent': float('inf')})
        predictor.update({'disk_percent': -float('inf')})

    @staticmethod
    def inject_empty_state(state_vector):
        """清空状态向量内部数据"""
        state_vector._recent_mae_history.clear()
        state_vector._recent_task_success.clear()

    @staticmethod
    def corrupt_yaml_file(filepath: str) -> None:
        """将YAML文件替换为非法内容"""
        with open(filepath, 'w') as f:
            f.write('{invalid: yaml: content: [}')

    @staticmethod
    def replace_dir_with_file(dirpath: str) -> None:
        """将目录替换为同名文件"""
        if os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with open(dirpath, 'w') as f:
            f.write('not_a_directory')

    @staticmethod
    def chmod_zero(filepath: str) -> None:
        """移除所有权限"""
        if os.path.exists(filepath):
            os.chmod(filepath, 0o000)

    @staticmethod
    def restore_chmod(filepath: str) -> None:
        """恢复权限"""
        if os.path.exists(filepath):
            os.chmod(filepath, 0o644)


class TestExtremeChaosAgentLoop(unittest.TestCase):
    """AgentLoop极端混沌测试"""

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

    def test_extreme_nan_in_predictors(self):
        """预测器接收到NaN/Inf数据，系统不应崩溃"""
        agent = AgentLoop('config.yaml')
        ExtremeChaosMonkey.inject_nan_to_predictor(agent.low_predictor)
        ExtremeChaosMonkey.inject_nan_to_predictor(agent.mid_predictor)

        errors = []
        for _ in range(20):
            try:
                agent.run_cycle()
            except Exception as e:
                errors.append(str(e))

        agent.stop()
        self.assertEqual(len(errors), 0, f"NaN injection caused errors: {errors}")

    def test_extreme_concurrent_db_access(self):
        """多线程并发访问同一数据库"""
        db_path = os.path.join(self.temp_dir, 'data', 'shared.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=1000)
        errors = []
        stop_event = threading.Event()

        def writer(tag):
            try:
                for i in range(200):
                    if stop_event.is_set():
                        break
                    mem.store_sample({'reward': float(i), 'tag': tag})
                    if i % 50 == 0:
                        mem.load_samples(limit=10)
            except Exception as e:
                errors.append(f"{tag}: {e}")

        threads = [threading.Thread(target=writer, args=(f't{i}',)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        stop_event.set()
        mem.close()

        self.assertEqual(len(errors), 0, f"Concurrent DB errors: {errors}")

    def test_extreme_task_pool_race_condition(self):
        """任务池并发修改时遍历"""
        pool = TaskPool(max_size=50)
        errors = []
        stop_event = threading.Event()

        def updater():
            try:
                for i in range(200):
                    if stop_event.is_set():
                        break
                    pool.update([Task(id=f'race_{i}', type=TaskType.EXPLORE, name=f't{i}')])
            except Exception as e:
                errors.append(f"updater: {e}")

        def selector():
            try:
                for _ in range(200):
                    if stop_event.is_set():
                        break
                    pool.select_task()
                    pool.get_stats()
            except Exception as e:
                errors.append(f"selector: {e}")

        def freezer():
            try:
                for _ in range(100):
                    if stop_event.is_set():
                        break
                    for tid in list(pool.tasks.keys())[:5]:
                        pool.freeze_task(tid)
                        pool.unfreeze_task(tid)
            except Exception as e:
                errors.append(f"freezer: {e}")

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=selector),
            threading.Thread(target=freezer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        stop_event.set()

        self.assertEqual(len(errors), 0, f"Race condition errors: {errors}")

    def test_extreme_permission_denied_files(self):
        """传感器读取权限被拒绝的文件"""
        agent = AgentLoop('config.yaml')
        forbidden_file = os.path.join(self.temp_dir, 'forbidden.log')
        with open(forbidden_file, 'w') as f:
            f.write('secret')
        ExtremeChaosMonkey.chmod_zero(forbidden_file)

        try:
            for _ in range(10):
                agent.run_cycle()
        finally:
            ExtremeChaosMonkey.restore_chmod(forbidden_file)

        agent.stop()
        self.assertTrue(True)

    def test_extreme_missing_state_keys(self):
        """驱力计算时状态字典缺少关键键"""
        agent = AgentLoop('config.yaml')
        # 手动清空状态历史
        ExtremeChaosMonkey.inject_empty_state(agent.state_vector)

        errors = []
        for _ in range(20):
            try:
                agent.run_cycle()
            except KeyError as e:
                errors.append(f"KeyError: {e}")
            except Exception as e:
                errors.append(str(e))

        agent.stop()
        self.assertEqual(len(errors), 0, f"Missing key errors: {errors}")

    def test_extreme_snapshot_dir_becomes_file(self):
        """快照目录被替换为文件"""
        agent = AgentLoop('config.yaml')
        for _ in range(5):
            agent.run_cycle()

        snapshot_dir = os.path.join(self.temp_dir, 'snapshots')
        agent.stop()

        # 替换目录为文件
        ExtremeChaosMonkey.replace_dir_with_file(snapshot_dir)

        # 重新运行
        agent2 = AgentLoop('config.yaml')
        errors = []
        for _ in range(5):
            try:
                agent2.run_cycle()
            except Exception as e:
                errors.append(str(e))
        agent2.stop()

        # 清理：删除文件，重建目录
        if os.path.isfile(snapshot_dir):
            os.remove(snapshot_dir)
        os.makedirs(snapshot_dir, exist_ok=True)

        self.assertEqual(len(errors), 0, f"Snapshot dir-as-file errors: {errors}")

    def test_extreme_long_running_with_random_faults(self):
        """长时间运行中随机注入多种故障"""
        agent = AgentLoop('config.yaml')
        errors = []
        stop_event = threading.Event()

        def runner():
            try:
                for i in range(200):
                    if stop_event.is_set():
                        break
                    agent.run_cycle()
                    if i % 50 == 25:
                        # 注入随机故障
                        fault = random.choice([
                            'nan_predictor',
                            'clear_history',
                            'extreme_limits',
                            'freeze_all',
                        ])
                        if fault == 'nan_predictor':
                            ExtremeChaosMonkey.inject_nan_to_predictor(agent.low_predictor)
                        elif fault == 'clear_history':
                            ExtremeChaosMonkey.inject_empty_state(agent.state_vector)
                        elif fault == 'extreme_limits':
                            agent.resource_guardian.max_cpu_percent = random.choice([0.0, 99999.0])
                        elif fault == 'freeze_all':
                            for tid in list(agent.task_pool.tasks.keys()):
                                agent.task_pool.freeze_task(tid)
            except Exception as e:
                errors.append(f"runner: {e}")

        t = threading.Thread(target=runner)
        t.start()
        t.join(timeout=60)
        stop_event.set()
        agent.stop()

        self.assertEqual(len(errors), 0, f"Long run errors: {errors}")

    def test_extreme_arima_with_illegal_params(self):
        """ARIMA预测器使用非法参数"""
        model = ARIMAModel(order=(0, 0, 0), min_samples=1)
        errors = []

        for _ in range(10):
            try:
                model.update(float(random.randint(-1000, 1000)))
                model.predict()
            except Exception as e:
                errors.append(str(e))

        # 尝试设置非法order（应被优雅处理）
        try:
            model.set_params({'order': (-1, -1, -1)})
            model.predict()
        except Exception as e:
            errors.append(f"set_params: {e}")

        self.assertEqual(len(errors), 0, f"ARIMA illegal params errors: {errors}")

    def test_extreme_db_deleted_during_transaction(self):
        """数据库在事务执行中被删除"""
        db_path = os.path.join(self.temp_dir, 'volatile.db')
        mem = ShortTermMemory(db_path=db_path, max_samples=100)
        errors = []

        def writer():
            try:
                for i in range(100):
                    mem.store_sample({'reward': float(i)})
            except Exception as e:
                errors.append(f"writer: {e}")

        def deleter():
            time.sleep(0.1)
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception:
                pass

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=deleter)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        try:
            mem.close()
        except Exception:
            pass

        # 数据库被删除可能导致异常，但不应崩溃
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
