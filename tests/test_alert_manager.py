import os
import sys
import unittest
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.alert_manager import AlertManager


class TestAlertManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.alert_log = os.path.join(self.temp_dir, 'alerts.jsonl')
        self.manager = AlertManager(alert_log_path=self.alert_log)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trigger_creates_alert(self):
        alert = self.manager.trigger('warning', 'test_source', 'test message')
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, 'warning')
        self.assertEqual(alert.source, 'test_source')
        self.assertEqual(alert.message, 'test message')

    def test_cooldown_blocks_duplicate(self):
        self.manager.trigger('warning', 'test_source', 'first')
        alert2 = self.manager.trigger('warning', 'test_source', 'second')
        self.assertIsNone(alert2)

    def test_different_source_no_cooldown(self):
        self.manager.trigger('warning', 'source_a', 'msg')
        alert2 = self.manager.trigger('warning', 'source_b', 'msg')
        self.assertIsNotNone(alert2)

    def test_get_active_alerts(self):
        self.manager.trigger('critical', 'safety', 'emergency')
        self.manager.trigger('info', 'system', 'info msg')
        active = self.manager.get_active_alerts()
        self.assertEqual(len(active), 2)

    def test_get_active_alerts_filter(self):
        self.manager.trigger('critical', 'safety', 'emergency')
        self.manager.trigger('warning', 'system', 'warn')
        critical = self.manager.get_active_alerts(severity_filter='critical')
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0]['severity'], 'critical')

    def test_acknowledge(self):
        alert = self.manager.trigger('warning', 'test', 'msg')
        self.assertTrue(self.manager.acknowledge(alert.alert_id))
        active = self.manager.get_active_alerts()
        self.assertEqual(len(active), 0)

    def test_acknowledge_not_found(self):
        self.assertFalse(self.manager.acknowledge('nonexistent'))

    def test_stats(self):
        self.manager.trigger('critical', 'a', 'm1')
        self.manager.trigger('warning', 'b', 'm2')
        stats = self.manager.get_stats()
        self.assertEqual(stats['total_alerts'], 2)
        self.assertEqual(stats['active_alerts'], 2)
        self.assertEqual(stats['critical_count'], 1)
        self.assertEqual(stats['warning_count'], 1)
        self.assertEqual(stats['info_count'], 0)

    def test_clear_old_alerts(self):
        self.manager.trigger('info', 'old', 'old msg')
        time.sleep(0.05)
        deleted = self.manager.clear_old_alerts(max_age_seconds=0.01)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.manager.get_stats()['total_alerts'], 0)

    def test_file_handler_writes(self):
        self.manager.trigger('warning', 'test', 'file test')
        self.assertTrue(os.path.exists(self.alert_log))
        with open(self.alert_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        import json
        record = json.loads(lines[0])
        self.assertEqual(record['severity'], 'warning')

    def test_custom_handler(self):
        received = []
        def custom_handler(alert):
            received.append(alert.alert_id)
        self.manager.add_handler(custom_handler)
        alert = self.manager.trigger('info', 'custom', 'custom msg')
        self.assertIn(alert.alert_id, received)


if __name__ == '__main__':
    unittest.main()
