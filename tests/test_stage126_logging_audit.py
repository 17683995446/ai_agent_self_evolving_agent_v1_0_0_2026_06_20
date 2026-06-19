"""阶段126：日志可追溯性与审计链路测试

验证关键操作日志可完整追溯，审计链路无断点。
暴露并修复以下问题：
1. AutoRollback 完全没有日志
2. SnapshotManager load/delete 缺少日志
3. 日志级别动态切换能力
4. 日志文件轮转
5. 磁盘满时日志写入保护
6. AlertManager 告警记录完整性
"""
import os
import sys
import json
import shutil
import tempfile
import logging
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.snapshot_manager import SnapshotManager
from src.optimization.auto_rollback import AutoRollback
from src.monitoring.alert_manager import AlertManager
from src.core.agent_loop import AgentLoop


class _LogTestBase(unittest.TestCase):
    """日志测试基类。"""

    def setUp(self):
        # 重置 Config 单例，避免测试间配置污染
        from src.core.config import Config
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def tearDown(self):
        from src.core.config import Config
        Config._instance = None
        Config._config = {}
        Config._singleton = True
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_agent(self, config_overrides=None):
        """创建临时配置的 AgentLoop。"""
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
                'level': 'INFO',
                'file': 'agent.log',
                'max_bytes': 1024,
                'backup_count': 2,
            },
            'paths': {
                'data_dir': self.temp_dir,
                'snapshot_dir': os.path.join(self.temp_dir, 'snapshots'),
                'log_dir': self.log_dir,
            },
            'auto_rollback': {'mae_degradation_threshold': 0.2},
            'task_pool': {'max_size': 50},
        }
        if config_overrides:
            config.update(config_overrides)
        import yaml
        config_path = os.path.join(self.temp_dir, 'config.yaml')
        yaml_content = yaml.dump(config, default_flow_style=False)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        agent = AgentLoop(config_path)
        return agent

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


# ============================================================
# Bug 1: AutoRollback 缺少日志
# ============================================================

class TestAutoRollbackLogging(_LogTestBase):
    """验证 AutoRollback 的关键操作有日志记录。"""

    def test_rollback_trigger_logs(self):
        """触发回滚时应记录日志。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        sm.max_count = 10
        ar = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=0.1)

        # 创建基线快照
        baseline_id = sm.create_snapshot({'state': 'baseline'}, tag='baseline')
        ar.set_baseline(mae=0.05, snapshot_id=baseline_id)

        # 使用低质量模型触发退化
        bad_model = {'state': 'bad'}
        sm.create_snapshot(bad_model, tag='test')

        # 退化检测窗口=3
        ar.check_and_rollback(current_mae=0.5)
        ar.check_and_rollback(current_mae=0.5)
        with self.assertLogs('src.optimization.auto_rollback', level='INFO') as cm:
            result = ar.check_and_rollback(current_mae=0.5)
        self.assertEqual(result.get('action'), 'rollback',
                         "应触发回滚")
        logs = '\n'.join(cm.output)
        self.assertIn('rollback', logs.lower(),
                      "回滚操作应记录到日志")

    def test_baseline_capture_logs(self):
        """捕获基线时应记录日志。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        ar = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=0.1)

        with self.assertLogs('src.optimization.auto_rollback', level='INFO') as cm:
            ar.set_baseline(mae=0.05, snapshot_id='snap_1')

        logs = '\n'.join(cm.output)
        self.assertIn('baseline', logs.lower(),
                      "捕获基线应记录到日志")


# ============================================================
# Bug 2: SnapshotManager load/delete 缺少日志
# ============================================================

class TestSnapshotManagerLogging(_LogTestBase):
    """验证 SnapshotManager 的 load/delete 操作有日志记录。"""

    def test_create_snapshot_logs(self):
        """创建快照时应记录日志。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        with self.assertLogs('src.optimization.snapshot_manager', level='DEBUG') as cm:
            sm.create_snapshot({'state': 1}, tag='test')
        logs = '\n'.join(cm.output)
        self.assertIn('create_snapshot', logs,
                      "创建快照应记录日志")

    def test_load_snapshot_logs(self):
        """加载快照时应记录日志。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        sid = sm.create_snapshot({'state': 1}, tag='test')

        with self.assertLogs('src.optimization.snapshot_manager', level='INFO') as cm:
            loaded = sm.load_snapshot(sid)

        self.assertEqual(loaded, {'state': 1})
        logs = '\n'.join(cm.output)
        self.assertIn(sid, logs,
                      "加载快照日志应包含快照ID")

    def test_load_missing_snapshot_logs(self):
        """加载不存在的快照时应记录日志。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)

        with self.assertLogs('src.optimization.snapshot_manager', level='WARNING') as cm:
            result = sm.load_snapshot('nonexistent')

        self.assertIsNone(result)
        logs = '\n'.join(cm.output)
        self.assertIn('nonexistent', logs,
                      "加载不存在的快照应记录警告日志")

    def test_delete_snapshot_logs(self):
        """删除快照时应记录日志。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        sid = sm.create_snapshot({'state': 1}, tag='test')

        with self.assertLogs('src.optimization.snapshot_manager', level='INFO') as cm:
            deleted = sm.delete_snapshot(sid)

        self.assertTrue(deleted)
        logs = '\n'.join(cm.output)
        self.assertIn(sid, logs,
                      "删除快照日志应包含快照ID")


# ============================================================
# Bug 3: 日志级别动态切换
# ============================================================

class TestLogLevelSwitching(_LogTestBase):
    """验证日志级别可以动态切换。"""

    def test_agent_loop_can_switch_log_level(self):
        """AgentLoop 应支持动态切换日志级别。"""
        agent = self._make_agent()
        try:
            # 初始级别 INFO
            root = logging.getLogger()
            self.assertEqual(root.level, logging.INFO)

            # 切换为 DEBUG
            agent._configure_root_logger(logging.DEBUG, agent._log_handler)
            self.assertEqual(root.level, logging.DEBUG)

            # 切换回 WARNING
            agent._configure_root_logger(logging.WARNING, agent._log_handler)
            self.assertEqual(root.level, logging.WARNING)
        finally:
            self._cleanup_agent(agent)

    def test_debug_logs_visible_after_level_switch(self):
        """切换为 DEBUG 后应能看到 debug 日志。"""
        agent = self._make_agent()
        try:
            agent._configure_root_logger(logging.DEBUG, agent._log_handler)
            with self.assertLogs(level='DEBUG') as cm:
                agent.run_cycle()
            logs = '\n'.join(cm.output)
            self.assertIn('STEP', logs,
                          "DEBUG 级别应能看到 cycle step 日志")
        finally:
            self._cleanup_agent(agent)


# ============================================================
# Bug 4: 日志文件轮转
# ============================================================

class TestLogRotation(_LogTestBase):
    """验证日志文件轮转机制。"""

    def test_log_rotation_creates_backup_files(self):
        """日志超过 maxBytes 后应创建备份文件。"""
        agent = self._make_agent({'logging': {'max_bytes': 50, 'backup_count': 2}})
        try:
            # 直接向 agent 的 handler 写入超过 maxBytes 的消息，触发轮转
            handler = agent._log_handler
            big_msg = 'x' * 100
            record = logging.LogRecord(
                name='rotation_test', level=logging.INFO,
                pathname='', lineno=0, msg=big_msg, args=(), exc_info=None
            )
            handler.emit(record)
            handler.emit(record)
            handler.flush()

            # 应该至少有 agent.log 和一个 backup
            files = os.listdir(self.log_dir)
            backups = [f for f in files if f.startswith('agent.log')]
            self.assertGreater(len(backups), 1,
                               "应创建日志备份文件")
        finally:
            self._cleanup_agent(agent)


# ============================================================
# Bug 5: 磁盘满时日志写入保护
# ============================================================

class TestLogDiskFullProtection(_LogTestBase):
    """验证磁盘满时日志系统不崩溃。"""

    def test_log_write_on_full_disk_does_not_crash(self):
        """模拟磁盘满，SafeRotatingFileHandler 应不抛出未捕获异常。"""
        from src.core.agent_loop import SafeRotatingFileHandler

        log_file = os.path.join(self.log_dir, 'safe.log')
        handler = SafeRotatingFileHandler(log_file, maxBytes=1024, backupCount=1)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            # Mock emit 内部抛出 OSError
            original_emit = logging.handlers.RotatingFileHandler.emit

            def failing_emit(self, record):
                raise OSError(28, "No space left on device")

            logging.handlers.RotatingFileHandler.emit = failing_emit
            try:
                root.info("This should not crash despite disk full")
            except Exception as e:
                self.fail(f"磁盘满时日志写入不应崩溃: {e}")
            finally:
                logging.handlers.RotatingFileHandler.emit = original_emit
        finally:
            root.removeHandler(handler)


# ============================================================
# AlertManager 审计记录
# ============================================================

class TestAlertManagerAudit(_LogTestBase):
    """验证 AlertManager 告警记录完整性。"""

    def test_alert_contains_required_fields(self):
        """告警对象应包含所有审计字段。"""
        alert_log = os.path.join(self.temp_dir, 'alerts.jsonl')
        am = AlertManager(alert_log_path=alert_log)
        alert = am.trigger('warning', 'test_source', 'test message')

        self.assertIsNotNone(alert)
        self.assertTrue(hasattr(alert, 'alert_id'))
        self.assertTrue(hasattr(alert, 'severity'))
        self.assertTrue(hasattr(alert, 'source'))
        self.assertTrue(hasattr(alert, 'message'))
        self.assertTrue(hasattr(alert, 'timestamp'))

    def test_alert_written_to_jsonl(self):
        """告警应写入 JSONL 文件。"""
        alert_log = os.path.join(self.temp_dir, 'alerts.jsonl')
        am = AlertManager(alert_log_path=alert_log)
        am.trigger('warning', 'test_source', 'test message')

        self.assertTrue(os.path.exists(alert_log))
        with open(alert_log, 'r', encoding='utf-8') as f:
            line = f.readline()
        record = json.loads(line)
        self.assertEqual(record['severity'], 'warning')
        self.assertEqual(record['source'], 'test_source')
        self.assertEqual(record['message'], 'test message')
        self.assertIn('timestamp', record)
        self.assertIn('alert_id', record)

    def test_alert_cooldown_prevents_duplicate(self):
        """冷却期内相同告警不应重复触发。"""
        am = AlertManager(alert_log_path=None)
        alert1 = am.trigger('warning', 'test_source', 'message1')
        alert2 = am.trigger('warning', 'test_source', 'message2')

        self.assertIsNotNone(alert1)
        self.assertIsNone(alert2,
                         "冷却期内相同来源和严重级别的告警不应重复触发")

    def test_alert_handler_exception_isolation(self):
        """单个告警处理器异常不应影响其他处理器。"""
        alert_log = os.path.join(self.temp_dir, 'alerts.jsonl')
        am = AlertManager(alert_log_path=alert_log)
        received = []

        def bad_handler(alert):
            raise RuntimeError("intentional error")

        def good_handler(alert):
            received.append(alert)

        am.add_handler(bad_handler)
        am.add_handler(good_handler)
        am.trigger('info', 'test', 'test message')

        self.assertEqual(len(received), 1,
                         "好处理器应正常接收到告警")


# ============================================================
# 端到端审计链路
# ============================================================

class TestEndToEndAuditTrail(_LogTestBase):
    """验证关键操作的端到端审计链路。"""

    def test_full_cycle_logs_all_steps(self):
        """一个完整 cycle 应记录所有关键步骤日志。"""
        agent = self._make_agent()
        try:
            with self.assertLogs(level='INFO') as cm:
                agent.run_cycle()
            logs = '\n'.join(cm.output)
            required_keywords = ['step', 'predict', 'evaluate', 'reward']
            for kw in required_keywords:
                self.assertIn(kw, logs.lower(),
                              f"完整 cycle 日志应包含 {kw}")
        finally:
            self._cleanup_agent(agent)

    def test_snapshot_creation_audit_trail(self):
        """创建快照应留下完整审计链路。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        with self.assertLogs('src.optimization.snapshot_manager', level='INFO') as cm:
            sid = sm.create_snapshot({'state': 'audit'}, tag='audit')

        logs = '\n'.join(cm.output)
        self.assertIn(sid, logs,
                      "快照创建审计日志应包含快照ID")

    def test_rollback_audit_trail(self):
        """回滚操作应留下完整审计链路。"""
        sm = SnapshotManager(snapshot_dir=self.temp_dir)
        ar = AutoRollback(snapshot_manager=sm, mae_degradation_threshold=0.1)
        baseline_id = sm.create_snapshot({'state': 'baseline'}, tag='baseline')
        ar.set_baseline(mae=0.05, snapshot_id=baseline_id)

        bad_model = {'state': 'bad'}
        sm.create_snapshot(bad_model, tag='test')

        # 触发退化窗口
        ar.check_and_rollback(current_mae=0.5)
        ar.check_and_rollback(current_mae=0.5)
        with self.assertLogs('src.optimization.auto_rollback', level='INFO') as cm:
            ar.check_and_rollback(current_mae=0.5)

        logs = '\n'.join(cm.output)
        self.assertIn('snapshot', logs.lower(),
                      "回滚日志应包含快照ID")


if __name__ == '__main__':
    unittest.main()
