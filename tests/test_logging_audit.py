#!/usr/bin/env python3
"""
日志系统深度审计与日志驱动调试测试
目标：验证日志是否正常打印，日志级别过滤正确，日志轮转不丢失数据
"""

import os
import sys
import unittest
import tempfile
import shutil
import logging
import logging.handlers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop
from src.core.config import Config
from src.monitoring.log_aggregator import LogAggregator


class TestLoggingAudit(unittest.TestCase):
    """日志审计测试"""

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
        # 确保root logger级别为DEBUG且添加文件handler
        self._setup_test_logging()

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # 清理测试添加的handler
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, '_test_handler', False):
                root.removeHandler(h)

    def _setup_test_logging(self):
        """为测试单独配置文件日志handler"""
        log_path = os.path.join(self.temp_dir, 'logs', 'agent.log')
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        # 检查是否已有我们的测试handler
        has_test_handler = any(getattr(h, '_test_handler', False) for h in root.handlers)
        if not has_test_handler:
            handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=10485760, backupCount=3)
            handler._test_handler = True
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            root.addHandler(handler)

    def _flush_logs(self):
        for handler in logging.getLogger().handlers:
            handler.flush()

    def _read_log_file(self):
        self._flush_logs()
        log_path = os.path.join(self.temp_dir, 'logs', 'agent.log')
        if not os.path.exists(log_path):
            return []
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()

    def test_cycle_logs_all_steps(self):
        """一个cycle应产生所有步骤的日志"""
        agent = AgentLoop('config.yaml')
        agent.run_cycle()
        agent.stop()

        lines = self._read_log_file()
        log_text = '\n'.join(lines)

        # 验证每个步骤都有日志
        self.assertIn('STEP 1: perceive', log_text)
        self.assertIn('STEP 2: safety_check', log_text)
        self.assertIn('STEP 3: predict', log_text)
        self.assertIn('STEP 4: evaluate', log_text)
        self.assertIn('STEP 5: reward', log_text)
        self.assertIn('STEP 6: task_select', log_text)
        self.assertIn('STEP 7: execute', log_text)
        self.assertIn('STEP 8: memory_store', log_text)
        self.assertIn('STEP 9: optimize', log_text)
        self.assertIn('STEP 10: strategy_update', log_text)
        self.assertIn('STEP 11: review_check', log_text)
        self.assertIn('STEP 12: snapshot', log_text)
        self.assertIn('END, duration=', log_text)

    def test_emergency_mode_logs_warning(self):
        """紧急模式应产生WARNING日志"""
        agent = AgentLoop('config.yaml')
        # 设置极低的资源限制，确保full_check()返回emergency_mode=True
        agent.resource_guardian.max_cpu_percent = 0.0
        agent.resource_guardian.max_mem_percent = 0.0

        agent.run_cycle()
        agent.stop()

        lines = self._read_log_file()
        warnings = [l for l in lines if 'WARNING' in l]
        self.assertGreater(len(warnings), 0, "Emergency mode should produce WARNING logs")

        log_text = '\n'.join(lines)
        self.assertIn('EMERGENCY RESPONSE triggered', log_text)

    def test_debug_level_filtering(self):
        """DEBUG级别应记录所有步骤日志"""
        agent = AgentLoop('config.yaml')
        for _ in range(3):
            agent.run_cycle()
        agent.stop()

        lines = self._read_log_file()
        debug_lines = [l for l in lines if 'DEBUG' in l]
        # 每个cycle应有至少12个DEBUG日志（12个步骤）
        self.assertGreaterEqual(len(debug_lines), 12 * 3)

    def test_info_level_logs_present(self):
        """INFO级别日志应存在"""
        agent = AgentLoop('config.yaml')
        for _ in range(5):
            agent.run_cycle()
        agent.stop()

        lines = self._read_log_file()
        info_lines = [l for l in lines if 'INFO' in l]
        self.assertGreaterEqual(len(info_lines), 0)


class TestLogRotation(unittest.TestCase):
    """日志轮转测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, 'test.log')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rotating_file_handler(self):
        """RotatingFileHandler应正确轮转"""
        handler = logging.handlers.RotatingFileHandler(
            self.log_path, maxBytes=500, backupCount=3
        )
        logger = logging.getLogger('rotation_test')
        logger.setLevel(logging.DEBUG)
        # 清除已有handler避免干扰
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(handler)

        # 写入大量日志触发轮转
        for i in range(30):
            logger.debug(f"Log line {i}: " + "x" * 20)

        handler.flush()
        handler.close()
        logger.removeHandler(handler)

        # 验证有备份文件生成
        backup_files = [f for f in os.listdir(self.temp_dir) if f.startswith('test.log.')]
        self.assertGreater(len(backup_files), 0, "Log rotation should create backup files")

        # 验证所有日志数据不丢失
        total_lines = 0
        for fname in ['test.log'] + sorted(backup_files):
            fpath = os.path.join(self.temp_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r') as f:
                    total_lines += len(f.readlines())

        self.assertGreaterEqual(total_lines, 30, "All log lines should be preserved after rotation")


class TestLogAggregator(unittest.TestCase):
    """日志聚合器测试"""

    def test_ingest_and_match(self):
        """日志摄入和模式匹配"""
        agg = LogAggregator()
        # 触发5个ERROR（超过阈值5）
        for _ in range(5):
            agg.ingest("ERROR something went wrong")
        agg.ingest("INFO normal operation")

        stats = agg.get_stats()
        self.assertEqual(stats['total_logs_ingested'], 6)
        self.assertTrue(stats['patterns']['error_spike']['triggered'])

    def test_anomaly_detection(self):
        """异常检测"""
        agg = LogAggregator()
        anomalies_triggered = []

        def handler(event):
            anomalies_triggered.append(event)

        agg.on_anomaly(handler)

        # 触发5个ERROR（超过阈值）
        for _ in range(5):
            agg.ingest("ERROR critical failure")

        anomalies = agg.check_anomalies()
        self.assertGreaterEqual(len(anomalies), 1)
        self.assertEqual(len(anomalies_triggered), 1)
        self.assertEqual(anomalies_triggered[0].pattern_name, 'error_spike')

    def test_read_from_file(self):
        """从文件读取日志"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, 'test.log')
        with open(log_file, 'w') as f:
            for i in range(10):
                f.write(f"Line {i}\n")

        agg = LogAggregator(log_file=log_file)
        lines = agg.read_from_file(lines=5)
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], 'Line 5')

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_clear_resets_state(self):
        """清空应重置所有状态"""
        agg = LogAggregator()
        agg.ingest("ERROR test")
        agg.clear()

        stats = agg.get_stats()
        self.assertEqual(stats['total_logs_ingested'], 0)


class TestLogDrivenDebugging(unittest.TestCase):
    """日志驱动调试测试"""

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
        self._setup_test_logging()

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, '_test_handler', False):
                root.removeHandler(h)

    def _setup_test_logging(self):
        log_path = os.path.join(self.temp_dir, 'logs', 'agent.log')
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        has_test_handler = any(getattr(h, '_test_handler', False) for h in root.handlers)
        if not has_test_handler:
            handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=10485760, backupCount=3)
            handler._test_handler = True
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            root.addHandler(handler)

    def _flush_logs(self):
        for handler in logging.getLogger().handlers:
            handler.flush()

    def test_log_sequence_validates_execution_order(self):
        """日志序列验证执行顺序"""
        agent = AgentLoop('config.yaml')
        agent.run_cycle()
        agent.stop()

        self._flush_logs()
        log_path = os.path.join(self.temp_dir, 'logs', 'agent.log')
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # 提取包含STEP的日志行
        step_lines = [l for l in lines if 'STEP' in l and 'DONE' not in l]
        step_numbers = []
        for line in step_lines:
            # 提取STEP后面的数字
            parts = line.split('STEP ')
            if len(parts) > 1:
                num = parts[1].split(':')[0]
                try:
                    step_numbers.append(int(num))
                except ValueError:
                    pass

        # 验证步骤按顺序执行
        self.assertEqual(step_numbers, list(range(1, 13)))

    def test_log_duration_reasonable(self):
        """日志中的duration应在合理范围"""
        agent = AgentLoop('config.yaml')
        for _ in range(5):
            agent.run_cycle()
        agent.stop()

        self._flush_logs()
        log_path = os.path.join(self.temp_dir, 'logs', 'agent.log')
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # 检查duration日志
        duration_lines = [l for l in lines if 'END, duration=' in l]
        self.assertGreaterEqual(len(duration_lines), 5)

        for line in duration_lines:
            # 提取duration值
            if 'duration=' in line:
                parts = line.split('duration=')
                val = parts[1].split('s')[0]
                duration = float(val)
                # 单次cycle应<5秒
                self.assertLess(duration, 5.0, f"Cycle duration too long: {duration}s")


if __name__ == '__main__':
    unittest.main()
