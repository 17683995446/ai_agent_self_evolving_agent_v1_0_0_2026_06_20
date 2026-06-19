import os
import sys
import unittest
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.log_aggregator import LogAggregator, LogPattern, AnomalyEvent


class TestLogPattern(unittest.TestCase):
    def test_pattern_creation(self):
        p = LogPattern('test', r'ERROR', 'warning', threshold=3, window_seconds=60.0)
        self.assertEqual(p.name, 'test')
        self.assertEqual(p.threshold, 3)


class TestAnomalyEvent(unittest.TestCase):
    def test_event_to_dict(self):
        e = AnomalyEvent('pattern1', 'critical', 'msg', 5)
        d = e.to_dict()
        self.assertEqual(d['pattern_name'], 'pattern1')
        self.assertEqual(d['severity'], 'critical')
        self.assertEqual(d['count'], 5)


class TestLogAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = LogAggregator()

    def test_ingest(self):
        self.aggregator.ingest("INFO: normal operation")
        self.assertEqual(len(self.aggregator._logs), 1)

    def test_ingest_batch(self):
        lines = ["line1", "line2", "line3"]
        self.aggregator.ingest_batch(lines)
        self.assertEqual(len(self.aggregator._logs), 3)

    def test_pattern_match(self):
        self.aggregator.ingest("ERROR: something failed")
        self.assertEqual(len(self.aggregator._matches['error_spike']), 1)

    def test_no_match(self):
        self.aggregator.ingest("INFO: everything ok")
        self.assertEqual(len(self.aggregator._matches['error_spike']), 0)

    def test_check_anomalies_no_trigger(self):
        self.aggregator.ingest("ERROR: one error")
        anomalies = self.aggregator.check_anomalies()
        # threshold=5, only 1 match
        self.assertEqual(len(anomalies), 0)

    def test_check_anomalies_triggered(self):
        for _ in range(5):
            self.aggregator.ingest("ERROR: something failed")

        anomalies = self.aggregator.check_anomalies()
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].pattern_name, 'error_spike')
        self.assertEqual(anomalies[0].severity, 'warning')

    def test_check_anomalies_multiple_patterns(self):
        for _ in range(5):
            self.aggregator.ingest("ERROR: failed")
        for _ in range(3):
            self.aggregator.ingest("Task execution failed")

        anomalies = self.aggregator.check_anomalies()
        names = [a.pattern_name for a in anomalies]
        self.assertIn('error_spike', names)
        self.assertIn('task_failure', names)

    def test_anomaly_handler(self):
        events = []

        def handler(event):
            events.append(event)

        self.aggregator.on_anomaly(handler)

        for _ in range(5):
            self.aggregator.ingest("ERROR: failed")

        self.aggregator.check_anomalies()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].pattern_name, 'error_spike')

    def test_read_from_file(self):
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, 'test.log')

        with open(log_file, 'w') as f:
            for i in range(20):
                f.write(f"Line {i}\n")

        aggregator = LogAggregator(log_file=log_file)
        lines = aggregator.read_from_file(lines=5)
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], "Line 15")
        self.assertEqual(lines[-1], "Line 19")

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_read_from_missing_file(self):
        aggregator = LogAggregator(log_file='/nonexistent/file.log')
        lines = aggregator.read_from_file()
        self.assertEqual(len(lines), 0)

    def test_get_stats(self):
        for _ in range(3):
            self.aggregator.ingest("ERROR: failed")

        stats = self.aggregator.get_stats()
        self.assertEqual(stats['total_logs_ingested'], 3)
        self.assertIn('error_spike', stats['patterns'])
        self.assertEqual(stats['patterns']['error_spike']['total_matches'], 3)

    def test_get_recent_logs(self):
        for i in range(10):
            self.aggregator.ingest(f"Log {i}")

        recent = self.aggregator.get_recent_logs(count=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0]['message'], "Log 7")
        self.assertEqual(recent[-1]['message'], "Log 9")

    def test_clear(self):
        self.aggregator.ingest("ERROR: test")
        self.aggregator.clear()
        self.assertEqual(len(self.aggregator._logs), 0)
        self.assertEqual(len(self.aggregator._matches['error_spike']), 0)

    def test_history_limit(self):
        aggregator = LogAggregator(history_size=5)
        for i in range(10):
            aggregator.ingest(f"Log {i}")

        self.assertEqual(len(aggregator._logs), 5)

    def test_window_filtering(self):
        # 创建一个自定义模式，窗口很小
        pattern = LogPattern('quick', r'QUICK', 'warning', threshold=2, window_seconds=0.1)
        aggregator = LogAggregator(patterns=[pattern])

        aggregator.ingest("QUICK event 1")
        aggregator.ingest("QUICK event 2")

        # 立即检查应该触发
        anomalies = aggregator.check_anomalies()
        self.assertEqual(len(anomalies), 1)

        # 等待窗口过期
        time.sleep(0.15)
        anomalies = aggregator.check_anomalies()
        self.assertEqual(len(anomalies), 0)

    def test_emergency_pattern(self):
        aggregator = LogAggregator()
        aggregator.ingest("Emergency mode activated")

        anomalies = aggregator.check_anomalies()
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].severity, 'critical')


if __name__ == '__main__':
    unittest.main()
