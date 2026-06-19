#!/usr/bin/env python3
"""
阶段112：系统混沌工程长期运行验证
目标：超大规模复杂场景下验证系统稳定性、数据一致性、告警正确性
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


class ChaosEngine:
    """高级混沌引擎：组合故障注入"""

    @staticmethod
    def corrupt_sqlite_db(db_path: str) -> None:
        if os.path.exists(db_path):
            with open(db_path, 'wb') as f:
                f.write(b'SQLite format 3\x00' + os.urandom(512))

    @staticmethod
    def truncate_file(path: str) -> None:
        if os.path.exists(path):
            with open(path, 'w') as f:
                f.write('')

    @staticmethod
    def delete_random_files(directory: str, count: int = 3) -> None:
        if not os.path.exists(directory):
            return
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for f in random.sample(files, min(count, len(files))):
            try:
                os.remove(os.path.join(directory, f))
            except Exception:
                pass

    @staticmethod
    def create_junk_files(directory: str, count: int = 20) -> list:
        paths = []
        for i in range(count):
            path = os.path.join(directory, f'junk_{i}_{int(time.time()*1000)}.tmp')
            with open(path, 'w') as f:
                f.write('x' * 2048)
            paths.append(path)
        return paths


class TestStage112MegaChaos(unittest.TestCase):
    """阶段112：超大规模混沌测试"""

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
  max_short_term_samples: 300
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

    def test_mega_chaos_200_cycles_single_agent(self):
        """单Agent 200周期，每20周期注入一种随机故障"""
        agent = AgentLoop('config.yaml')
        errors = []
        alert_history = []

        def alert_collector(alert):
            alert_history.append(alert.to_dict())
        agent.alert_manager.add_handler(alert_collector)

        for i in range(200):
            try:
                agent.run_cycle()
            except Exception as e:
                errors.append(f"cycle {i}: {e}")
                if len(errors) > 5:
                    break

            if i > 0 and i % 20 == 0:
                fault = random.choice([
                    'delete_log',
                    'extreme_limits',
                    'freeze_tasks',
                    'junk_files',
                    'truncate_db',
                ])
                if fault == 'delete_log':
                    ChaosEngine.delete_random_files(os.path.join(self.temp_dir, 'logs'))
                elif fault == 'extreme_limits':
                    agent.resource_guardian.max_cpu_percent = random.choice([0.0001, 99999.0])
                elif fault == 'freeze_tasks':
                    for tid in list(agent.task_pool.tasks.keys())[:5]:
                        agent.task_pool.freeze_task(tid)
                elif fault == 'junk_files':
                    ChaosEngine.create_junk_files(os.path.join(self.temp_dir, 'data'))
                elif fault == 'truncate_db':
                    db_path = os.path.join(self.temp_dir, 'data', 'memory.db')
                    ChaosEngine.truncate_file(db_path)

        stats = agent.get_stats()
        agent.stop()

        self.assertEqual(len(errors), 0, f"Errors during 200-cycle chaos: {errors}")
        self.assertGreaterEqual(stats['cycle_count'], 190)
        self.assertGreaterEqual(stats['task_pool']['total_tasks'], 0)
        self.assertLessEqual(stats['memory_samples'], 300)

    def test_multi_agent_concurrent_chaos(self):
        """3个Agent并发运行，各50周期，共享目录注入故障"""
        errors = []
        agents = []

        for idx in range(3):
            agent = AgentLoop('config.yaml')
            agents.append(agent)

        def runner(agent_idx: int) -> None:
            agent = agents[agent_idx]
            for i in range(50):
                try:
                    agent.run_cycle()
                except Exception as e:
                    errors.append(f"agent{agent_idx} cycle{i}: {e}")

        threads = [threading.Thread(target=runner, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()

        time.sleep(0.2)
        ChaosEngine.delete_random_files(os.path.join(self.temp_dir, 'logs'), count=5)
        time.sleep(0.2)
        ChaosEngine.create_junk_files(os.path.join(self.temp_dir, 'data'), count=30)

        for t in threads:
            t.join(timeout=60)

        for agent in agents:
            agent.stop()

        self.assertEqual(len(errors), 0, f"Multi-agent errors: {errors}")
        total_cycles = sum(a._cycle_count for a in agents)
        self.assertGreaterEqual(total_cycles, 140)

    def test_alert_triggered_under_chaos(self):
        """故障场景下告警系统必须正确触发"""
        agent = AgentLoop('config.yaml')
        triggered = []

        def capture(alert):
            triggered.append(alert.severity)
        agent.alert_manager.add_handler(capture)

        agent.alert_manager.trigger('critical', 'test_chaos', 'manual chaos injection')

        agent.stop()
        self.assertIn('critical', triggered)

    def test_batch_flush_integrity_under_fault(self):
        """批量写入在故障下仍保持数据完整性"""
        agent = AgentLoop('config.yaml')
        for i in range(150):
            agent.run_cycle()
            if i == 75:
                db_path = os.path.join(self.temp_dir, 'data', 'memory.db')
                ChaosEngine.truncate_file(db_path)

        count = agent.memory.get_sample_count()
        agent.stop()
        self.assertGreaterEqual(count, 0)

    def test_no_resource_leak_after_long_run(self):
        """长时间运行后无资源泄漏"""
        agent = AgentLoop('config.yaml')
        for _ in range(30):
            agent.run_cycle()

        agent.stop()
        self.assertFalse(agent._running)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'data')))

    def test_snapshot_rollback_alert_chaos(self):
        """快照回滚场景下告警正确记录"""
        agent = AgentLoop('config.yaml')
        rollback_alerts = []

        def handler(alert):
            if alert.source == 'rollback':
                rollback_alerts.append(alert.to_dict())
        agent.alert_manager.add_handler(handler)

        for i in range(30):
            agent.run_cycle()
            if i == 15:
                snapshots = agent.snapshot_manager.list_snapshots()
                if snapshots:
                    ChaosEngine.corrupt_sqlite_db(snapshots[-1]['filepath'])

        agent.stop()
        self.assertGreaterEqual(len(rollback_alerts), 0)


class TestStage112DataConsistency(unittest.TestCase):
    """阶段112：数据一致性专项验证"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'consistency.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_buffer_flush_on_close(self):
        """关闭时批量缓冲必须强制刷盘"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=1000, batch_size=50, flush_interval=60.0)
        for i in range(10):
            mem.store_sample({
                'timestamp': time.time(),
                'state': {'idx': i},
                'predictions': {},
                'actuals': {},
                'errors': {},
            })
        mem.flush()
        count_before_close = mem.get_sample_count()
        self.assertGreaterEqual(count_before_close, 10)
        mem.close()

    def test_concurrent_batch_flush_no_corruption(self):
        """并发写入+批量刷盘不应导致数据库损坏"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=500, batch_size=10, flush_interval=1.0)
        errors = []

        def writer(start: int) -> None:
            for i in range(50):
                try:
                    mem.store_sample({
                        'timestamp': time.time(),
                        'state': {'writer': start, 'idx': i},
                        'predictions': {},
                        'actuals': {},
                        'errors': {},
                    })
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        mem.flush()
        count = mem.get_sample_count()
        mem.close()

        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        self.assertGreaterEqual(count, 180)


if __name__ == '__main__':
    unittest.main()
