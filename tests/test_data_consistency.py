#!/usr/bin/env python3
"""
数据一致性校验与存储层加固测试
目标：验证数据流转无误，存储与回显完全匹配
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.short_term import ShortTermMemory
from src.memory.value_scorer import ValueScorer
from src.optimization.snapshot_manager import SnapshotManager


class TestShortTermMemoryConsistency(unittest.TestCase):
    """短期记忆数据一致性测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_load_roundtrip(self):
        """存储后加载，数据完全一致"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        original = {
            'timestamp': 1234567890.5,
            'state': {'system': {'cpu_percent': 50.0}},
            'predictions': {'low': {'cpu': 0.3}},
            'actuals': {'cpu': 0.35},
            'errors': {'cpu': 0.05},
            'task_id': 't1',
            'reward': 0.8,
            'value_score': 0.5,
        }
        sid = mem.store_sample(original)
        self.assertGreater(sid, 0)

        loaded = mem.load_samples(limit=1)[0]
        self.assertEqual(loaded['timestamp'], 1234567890.5)
        self.assertEqual(loaded['task_id'], 't1')
        self.assertEqual(loaded['reward'], 0.8)
        self.assertEqual(loaded['state'], {'system': {'cpu_percent': 50.0}})
        self.assertEqual(loaded['predictions'], {'low': {'cpu': 0.3}})
        self.assertEqual(loaded['actuals'], {'cpu': 0.35})
        self.assertEqual(loaded['errors'], {'cpu': 0.05})

    def test_load_samples_has_mae_field(self):
        """加载的样本必须包含mae字段供ValueScorer使用"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        mem.store_sample({
            'errors': {'cpu': 0.1, 'mem': 0.3},
            'reward': 0.5,
        })
        loaded = mem.load_samples(limit=1)[0]
        self.assertIn('mae', loaded)
        self.assertAlmostEqual(loaded['mae'], 0.2, places=6)

    def test_value_scorer_works_with_loaded_samples(self):
        """ValueScorer能正确计算从DB加载的样本分数"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        scorer = ValueScorer()

        mem.store_sample({
            'timestamp': time.time(),
            'errors': {'cpu': 0.8, 'mem': 0.9},
            'reward': 0.5,
        })
        loaded = mem.load_samples(limit=1)[0]
        score = scorer.compute_score(loaded)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # mae高应该导致分数高（error_weight=0.6）
        self.assertGreater(score, 0.3)

    def test_cleanup_retains_high_value_samples(self):
        """清理后高价值样本应被保留"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=10, batch_size=1, flush_interval=0.0)
        for i in range(20):
            mem.store_sample({
                'reward': float(i),
                'value_score': float(i) / 20.0,
                'errors': {'cpu': 0.1},
            })
        deleted = mem.cleanup_old_samples(retention_ratio=0.5)
        self.assertGreater(deleted, 0)
        count = mem.get_sample_count()
        self.assertEqual(count, 10)
        mem.close()

    def test_transaction_rollback_on_error(self):
        """事务异常时数据不应损坏"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        # 存储正常样本
        mem.store_sample({'reward': 1.0})
        count_before = mem.get_sample_count()

        # 模拟异常：通过关闭连接后强制操作（会被_ensure_open拦截）
        mem.close()
        sid = mem.store_sample({'reward': 2.0})
        self.assertEqual(sid, -1)

        # 重新打开验证之前的数据完好
        mem2 = ShortTermMemory(db_path=self.db_path, max_samples=100)
        count_after = mem2.get_sample_count()
        self.assertEqual(count_before, count_after)
        mem2.close()

    def test_json_serialization_consistency(self):
        """JSON序列化/反序列化一致性"""
        mem = ShortTermMemory(db_path=self.db_path, max_samples=100, batch_size=1, flush_interval=0.0)
        complex_data = {
            'state': {
                'nested': {'a': [1, 2, 3], 'b': None, 'c': True},
                'unicode': '中文测试',
            },
            'errors': {'cpu': float('inf'), 'mem': float('-inf')},
        }
        mem.store_sample(complex_data)
        loaded = mem.load_samples(limit=1)[0]
        self.assertEqual(loaded['state']['nested']['a'], [1, 2, 3])
        self.assertIsNone(loaded['state']['nested']['b'])
        self.assertTrue(loaded['state']['nested']['c'])
        self.assertEqual(loaded['state']['unicode'], '中文测试')


class TestSnapshotManagerConsistency(unittest.TestCase):
    """快照管理器数据一致性测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_load_roundtrip(self):
        """创建快照后加载，数据一致"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        model_state = {'cycle': 42, 'weights': [0.1, 0.2, 0.3]}
        sid = sm.create_snapshot(model_state, tag='test', metrics={'mae': 0.05})

        loaded = sm.load_snapshot(sid)
        self.assertEqual(loaded['cycle'], 42)
        self.assertEqual(loaded['weights'], [0.1, 0.2, 0.3])

    def test_index_persists_after_reload(self):
        """重启后index.json应正确恢复"""
        sm1 = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        sid = sm1.create_snapshot({'data': 1}, metrics={'mae': 0.1})

        sm2 = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        self.assertEqual(len(sm2.list_snapshots()), 1)
        loaded = sm2.load_snapshot(sid)
        self.assertEqual(loaded['data'], 1)

    def test_verify_index_removes_missing_files(self):
        """校验应移除index中记录但磁盘不存在的快照"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        sid = sm.create_snapshot({'data': 1})
        # 手动删除磁盘文件
        entry = sm.list_snapshots()[0]
        os.remove(entry['filepath'])

        # 重新加载应自动清理无效记录
        sm2 = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        self.assertEqual(len(sm2.list_snapshots()), 0)

    def test_verify_and_repair_cleans_orphans(self):
        """verify_and_repair应清理孤立文件"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        sm.create_snapshot({'data': 1})

        # 创建孤立文件
        orphan = os.path.join(self.temp_dir, 'orphan.pkl')
        with open(orphan, 'wb') as f:
            import pickle
            pickle.dump({}, f)

        result = sm.verify_and_repair()
        self.assertEqual(result['orphaned_files_removed'], 1)
        self.assertFalse(os.path.exists(orphan))

    def test_atomic_index_write(self):
        """index写入应是原子的，崩溃不会导致损坏"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        for i in range(5):
            sm.create_snapshot({'idx': i})

        # 验证index.json存在且有效
        index_path = os.path.join(self.temp_dir, 'index.json')
        self.assertTrue(os.path.exists(index_path))
        with open(index_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 5)

    def test_enforce_limit_consistency(self):
        """超限删除后index与磁盘保持一致"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=3)
        sids = []
        for i in range(5):
            sid = sm.create_snapshot({'idx': i})
            sids.append(sid)

        self.assertEqual(len(sm.list_snapshots()), 3)
        # 最早的2个应已被删除
        self.assertIsNone(sm.load_snapshot(sids[0]))
        self.assertIsNone(sm.load_snapshot(sids[1]))
        # 验证磁盘上也没有残留
        for sid in sids[:2]:
            filepath = os.path.join(self.temp_dir, f"{sid}.pkl")
            self.assertFalse(os.path.exists(filepath))


import time

if __name__ == '__main__':
    unittest.main()
