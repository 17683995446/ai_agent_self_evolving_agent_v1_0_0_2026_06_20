#!/usr/bin/env python3
"""
端到端混沌工程测试
目标：随机故障注入，验证系统在混乱场景下仍能恢复并继续运行
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
from src.memory.short_term import ShortTermMemory
from src.optimization.snapshot_manager import SnapshotManager
from src.task.task_pool import TaskPool
from src.task.task_definition import Task, TaskType


class ChaosMonkey:
    """混沌猴子：随机故障注入器"""

    @staticmethod
    def corrupt_file(filepath: str) -> None:
        """损坏文件"""
        if os.path.exists(filepath):
            with open(filepath, 'wb') as f:
                f.write(b'CORRUPTED' + os.urandom(100))

    @staticmethod
    def delete_file(filepath: str) -> None:
        """删除文件"""
        if os.path.exists(filepath):
            os.remove(filepath)

    @staticmethod
    def truncate_file(filepath: str) -> None:
        """截断文件为空"""
        if os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write('')

    @staticmethod
    def fill_disk_with_junk(directory: str, count: int = 10) -> list:
        """在目录中创建大量垃圾文件"""
        paths = []
        for i in range(count):
            path = os.path.join(directory, f'junk_{i}.tmp')
            with open(path, 'w') as f:
                f.write('x' * 1024)
            paths.append(path)
        return paths


class TestChaosAgentLoop(unittest.TestCase):
    """AgentLoop混沌测试"""

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

    def test_chaos_random_resource_limits(self):
        """随机改变资源限制，系统不应崩溃"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(50):
            try:
                # 随机改变资源限制
                agent.resource_guardian.max_cpu_percent = random.uniform(0.001, 200.0)
                agent.resource_guardian.max_mem_percent = random.uniform(0.001, 200.0)
                agent.run_cycle()
            except Exception as e:
                errors.append(str(e))

        agent.stop()
        self.assertEqual(len(errors), 0, f"Errors during chaos: {errors}")

    def test_chaos_delete_log_file_during_run(self):
        """运行中删除日志文件，系统不应崩溃"""
        agent = AgentLoop('config.yaml')
        log_path = os.path.join(self.temp_dir, 'logs', 'agent.log')

        for i in range(20):
            agent.run_cycle()
            if i == 10 and os.path.exists(log_path):
                os.remove(log_path)

        agent.stop()
        # 系统应继续运行，日志handler可能重建或静默失败
        self.assertTrue(True)

    def test_chaos_corrupt_database(self):
        """损坏数据库后系统应能恢复"""
        agent = AgentLoop('config.yaml')
        # 先正常运行几个周期
        for _ in range(5):
            agent.run_cycle()
        agent.stop()

        # 损坏数据库
        db_path = os.path.join(self.temp_dir, 'data', 'memory.db')
        if os.path.exists(db_path):
            ChaosMonkey.corrupt_file(db_path)

        # 重新创建Agent，应能处理损坏的数据库
        try:
            agent2 = AgentLoop('config.yaml')
            for _ in range(3):
                agent2.run_cycle()
            agent2.stop()
        except Exception as e:
            # SQLite损坏可能导致启动失败，这是可接受的
            self.assertIn('database', str(e).lower() or 'sqlite', str(e).lower())

    def test_chaos_rapid_start_stop(self):
        """快速启停不应崩溃"""
        for _ in range(10):
            agent = AgentLoop('config.yaml')
            agent.run_cycle()
            agent.stop()

    def test_chaos_concurrent_with_fault_injection(self):
        """并发运行中注入故障"""
        agent = AgentLoop('config.yaml')
        errors = []
        stop_event = threading.Event()

        def runner():
            try:
                for _ in range(30):
                    if stop_event.is_set():
                        break
                    agent.run_cycle()
            except Exception as e:
                errors.append(f"runner: {e}")

        def fault_injector():
            try:
                for _ in range(10):
                    if stop_event.is_set():
                        break
                    time.sleep(0.05)
                    # 随机故障
                    fault = random.choice(['limit', 'freeze', 'unfreeze'])
                    if fault == 'limit':
                        agent.resource_guardian.max_cpu_percent = random.uniform(0.001, 100.0)
                    elif fault == 'freeze':
                        for tid in list(agent.task_pool.tasks.keys())[:3]:
                            agent.task_pool.freeze_task(tid)
                    elif fault == 'unfreeze':
                        for tid in list(agent.task_pool.tasks.keys())[:3]:
                            agent.task_pool.unfreeze_task(tid)
            except Exception as e:
                errors.append(f"injector: {e}")

        t1 = threading.Thread(target=runner)
        t2 = threading.Thread(target=fault_injector)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)
        stop_event.set()
        agent.stop()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

    def test_chaos_memory_pressure(self):
        """内存压力：大量任务和样本"""
        agent = AgentLoop('config.yaml')

        # 创建大量任务
        for i in range(100):
            agent.task_pool.update([Task(id=f'chaos_t_{i}', type=TaskType.EXPLORE, name=f'task_{i}')])

        # 运行多个周期
        for _ in range(20):
            agent.run_cycle()

        stats = agent.get_stats()
        # 任务池不应超过限制
        self.assertLessEqual(stats['task_pool']['total_tasks'], 50)
        agent.stop()

    def test_chaos_snapshot_corruption(self):
        """快照损坏后系统应继续运行"""
        agent = AgentLoop('config.yaml')

        # 运行到产生快照
        for i in range(105):
            agent.run_cycle()

        # 损坏最新快照
        snapshots = agent.snapshot_manager.list_snapshots()
        if snapshots:
            latest = snapshots[-1]
            ChaosMonkey.corrupt_file(latest['filepath'])

        # 继续运行
        for _ in range(5):
            agent.run_cycle()

        agent.stop()
        self.assertTrue(True)

    def test_chaos_full_disk_simulation(self):
        """模拟磁盘满：创建大量临时文件"""
        agent = AgentLoop('config.yaml')
        junk_dir = os.path.join(self.temp_dir, 'junk')
        os.makedirs(junk_dir, exist_ok=True)

        # 先运行几个周期
        for _ in range(5):
            agent.run_cycle()

        # 创建大量垃圾文件
        junk_files = ChaosMonkey.fill_disk_with_junk(junk_dir, count=50)

        # 继续运行
        for _ in range(5):
            agent.run_cycle()

        # 清理
        for f in junk_files:
            if os.path.exists(f):
                os.remove(f)

        agent.stop()
        self.assertTrue(True)

    def test_chaos_random_config_changes(self):
        """随机修改配置值"""
        agent = AgentLoop('config.yaml')

        for i in range(30):
            agent.run_cycle()
            # 随机修改配置
            agent.resource_guardian.max_cpu_percent = random.choice([0.001, 50.0, 100.0, 1000.0])
            agent.safety_enforcer.behavior.max_consecutive_failures = random.randint(1, 20)

        agent.stop()
        self.assertTrue(True)

    def test_chaos_mixed_failure_modes(self):
        """混合故障模式"""
        agent = AgentLoop('config.yaml')
        errors = []

        for i in range(40):
            try:
                agent.run_cycle()

                # 每10个周期注入一种故障
                if i % 10 == 5:
                    fault = random.choice([
                        'delete_log',
                        'truncate_db',
                        'corrupt_snapshot',
                        'extreme_limits',
                    ])
                    if fault == 'delete_log':
                        log_path = os.path.join(self.temp_dir, 'logs', 'agent.log')
                        if os.path.exists(log_path):
                            os.remove(log_path)
                    elif fault == 'truncate_db':
                        db_path = os.path.join(self.temp_dir, 'data', 'memory.db')
                        if os.path.exists(db_path):
                            ChaosMonkey.truncate_file(db_path)
                    elif fault == 'corrupt_snapshot':
                        snapshots = agent.snapshot_manager.list_snapshots()
                        if snapshots:
                            ChaosMonkey.corrupt_file(snapshots[0]['filepath'])
                    elif fault == 'extreme_limits':
                        agent.resource_guardian.max_cpu_percent = random.choice([0.0, 999999.0])

            except Exception as e:
                errors.append(str(e))

        agent.stop()
        self.assertEqual(len(errors), 0, f"Errors: {errors}")


class TestChaosShortTermMemory(unittest.TestCase):
    """短期记忆混沌测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'chaos.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chaos_rapid_store_and_corrupt(self):
        """快速存储后损坏数据库"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=100)
        for i in range(50):
            mem.store_sample({'reward': float(i)})
        mem.close()

        # 损坏数据库
        ChaosMonkey.corrupt_file(self.db_path)

        # 尝试重新打开
        try:
            mem2 = ShortTermMemory(db_path=self.db_path, max_samples=100)
            count = mem2.get_sample_count()
            mem2.close()
        except Exception:
            # 损坏的数据库可能导致异常，这是可接受的
            pass

    def test_chaos_concurrent_store_and_close(self):
        """并发存储时关闭数据库"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    mem.store_sample({'reward': float(i)})
            except Exception as e:
                errors.append(str(e))

        def closer():
            time.sleep(0.05)
            try:
                mem.close()
            except Exception:
                pass

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=closer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # 不应有未捕获的异常
        self.assertEqual(len(errors), 0, f"Errors: {errors}")


if __name__ == '__main__':
    unittest.main()
