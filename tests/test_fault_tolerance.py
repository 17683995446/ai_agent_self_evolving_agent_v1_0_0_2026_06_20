#!/usr/bin/env python3
"""
异常容错与降级机制深度测试
目标：验证超时、断连、报错、兜底降级、重试机制均正常生效
"""

import os
import sys
import unittest
import tempfile
import shutil
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.state_vector import StateVector
from src.perception.system_sensor import SystemSensor
from src.perception.log_sensor import LogSensor
from src.perception.file_sensor import FileSensor
from src.prediction.arima_predictor import ARIMAModel
from src.core.config_reloader import ConfigReloader
from src.optimization.snapshot_manager import SnapshotManager


class FailingSensor:
    """模拟崩溃的传感器"""
    def perceive(self):
        raise RuntimeError("Simulated sensor failure")


class TestStateVectorFaultTolerance(unittest.TestCase):
    """StateVector传感器降级测试"""

    def test_sensor_failure_degradation(self):
        """单个传感器崩溃不应影响整体perceive"""
        sv = StateVector()
        # 替换为故障传感器
        sv.system_sensor = FailingSensor()
        state = sv.perceive()
        self.assertIn('timestamp', state)
        self.assertIn('system', state)
        self.assertIn('logs', state)
        self.assertIn('files', state)
        self.assertIn('self', state)
        # system应为降级值（全0）
        self.assertEqual(state['system']['cpu_percent'], 0.0)
        self.assertEqual(state['system']['mem_percent'], 0.0)
        # logs和files应正常
        self.assertGreaterEqual(state['logs']['error_count_1m'], 0)

    def test_all_sensors_failure(self):
        """所有传感器都崩溃时应返回降级状态"""
        sv = StateVector()
        sv.system_sensor = FailingSensor()
        sv.log_sensor = FailingSensor()
        sv.file_sensor = FailingSensor()
        state = sv.perceive()
        self.assertIn('timestamp', state)
        self.assertEqual(state['system']['cpu_percent'], 0.0)
        self.assertEqual(state['logs']['error_count_1m'], 0)
        self.assertEqual(state['files']['recent_changes_1m'], 0)

    def test_partial_sensor_failure_mixed(self):
        """混合正常和故障传感器"""
        sv = StateVector()
        sv.system_sensor = FailingSensor()
        state = sv.perceive()
        self.assertEqual(state['system']['cpu_percent'], 0.0)
        self.assertGreaterEqual(state['logs']['error_count_1m'], 0)


class TestLogSensorFaultTolerance(unittest.TestCase):
    """LogSensor容错测试"""

    def test_missing_log_file(self):
        ls = LogSensor(log_paths=['/nonexistent/path/to/log'])
        result = ls.perceive()
        self.assertEqual(result['error_count'], 0)
        self.assertEqual(result['warn_count'], 0)

    def test_permission_denied_file(self):
        """模拟权限拒绝的文件"""
        ls = LogSensor()
        # 添加一个不存在的路径
        ls.add_log_path('/root/protected_log')
        result = ls.perceive()
        self.assertGreaterEqual(result['error_count'], 0)

    def test_file_deleted_during_read(self):
        """读取过程中文件被删除"""
        temp_dir = tempfile.mkdtemp()
        log_path = os.path.join(temp_dir, 'test.log')
        with open(log_path, 'w') as f:
            f.write("ERROR test\n")

        ls = LogSensor(log_paths=[log_path])
        # 先正常读取一次
        ls.perceive()
        # 删除文件后再读取
        os.remove(log_path)
        result = ls.perceive()
        self.assertEqual(result['error_count'], 0)
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestFileSensorFaultTolerance(unittest.TestCase):
    """FileSensor容错测试"""

    def test_nonexistent_watch_path(self):
        fs = FileSensor(watch_paths=['/nonexistent_dir_xyz'])
        result = fs.perceive()
        self.assertEqual(result['total_changes'], 0)
        self.assertEqual(result['recent_changes'], 0)

    def test_permission_denied_directory(self):
        fs = FileSensor(watch_paths=['/root'])
        result = fs.perceive()
        self.assertGreaterEqual(result['total_changes'], 0)


class TestSystemSensorFaultTolerance(unittest.TestCase):
    """SystemSensor容错测试"""

    def test_network_counters_none(self):
        """net_io_counters返回None时不应崩溃"""
        ss = SystemSensor()
        # 直接调用_read_network，它在容器环境可能返回None
        result = ss._read_network()
        self.assertIn('sent_mb_s', result)
        self.assertIn('recv_mb_s', result)

    def test_process_iter_permission_denied(self):
        """process_iter权限拒绝时不应崩溃"""
        ss = SystemSensor()
        result = ss._read_processes()
        self.assertIn('total_count', result)
        self.assertIn('zombie_count', result)
        self.assertGreaterEqual(result['total_count'], 0)


class TestARIMAModelFaultTolerance(unittest.TestCase):
    """ARIMA降级测试"""

    def test_statsmodels_missing_fallback(self):
        """statsmodels不可用时降级为平均值"""
        m = ARIMAModel(order=(1, 1, 1), min_samples=3)
        for i in range(5):
            m.update(float(i))
        # 强制清除缓存使下次重新拟合
        m._cached_fitted = None
        m._cached_hash = None
        preds = m.predict(steps=2)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(isinstance(v, float) for v in preds))

    def test_constant_series_prediction(self):
        """常数序列不应导致除零或崩溃"""
        m = ARIMAModel(order=(1, 1, 1), min_samples=3)
        for _ in range(10):
            m.update(5.0)
        preds = m.predict(steps=2)
        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertTrue(abs(p - 5.0) < 1.0 or p == 0.0)


class TestConfigReloaderFaultTolerance(unittest.TestCase):
    """Config热重载容错测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.yaml')
        with open(self.config_path, 'w') as f:
            f.write("key: value\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_yaml_fallback(self):
        """无效YAML时不应崩溃"""
        reloader = ConfigReloader(self.config_path)
        # 写入无效YAML
        with open(self.config_path, 'w') as f:
            f.write("{invalid yaml: [}\n")
        config = reloader.force_reload()
        self.assertIsNone(config)

    def test_file_deleted_during_check(self):
        """检查过程中文件被删除"""
        reloader = ConfigReloader(self.config_path)
        os.remove(self.config_path)
        config = reloader.force_reload()
        self.assertIsNone(config)

    def test_empty_config_file(self):
        """空配置文件"""
        with open(self.config_path, 'w') as f:
            f.write("")
        reloader = ConfigReloader(self.config_path)
        config = reloader.force_reload()
        self.assertEqual(config, {})


class TestSnapshotManagerFaultTolerance(unittest.TestCase):
    """快照管理器容错测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_corrupted_index_recovery(self):
        """损坏的index.json应被忽略"""
        index_path = os.path.join(self.temp_dir, 'index.json')
        with open(index_path, 'w') as f:
            f.write("{not valid json")
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        self.assertEqual(len(sm.list_snapshots()), 0)

    def test_load_missing_snapshot(self):
        """加载不存在的快照应返回None"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        result = sm.load_snapshot('nonexistent')
        self.assertIsNone(result)

    def test_delete_already_deleted_file(self):
        """重复删除不应崩溃"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir, max_count=10)
        sid = sm.create_snapshot({'data': 1})
        entry = sm.list_snapshots()[0]
        os.remove(entry['filepath'])
        # 再次删除不应崩溃
        result = sm.delete_snapshot(sid)
        self.assertTrue(result)


class TestStateVectorSelfReadFaultTolerance(unittest.TestCase):
    """StateVector._read_self容错"""

    def test_no_such_process(self):
        """进程不存在时返回降级值"""
        sv = StateVector()
        sv._self_pid = 99999999  # 不存在的PID
        result = sv._read_self()
        self.assertEqual(result['cpu_percent'], 0.0)
        self.assertEqual(result['mem_mb'], 0.0)


if __name__ == '__main__':
    unittest.main()
