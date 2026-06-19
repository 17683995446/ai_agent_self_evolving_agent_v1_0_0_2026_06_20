import os
import sys
import unittest
import tempfile
import shutil
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.daemon import DaemonManager, run_daemon_command


class TestDaemonManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pid_file = os.path.join(self.temp_dir, 'test.pid')

    def tearDown(self):
        # 清理可能残留的PID文件（不kill当前测试进程）
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                if pid != os.getpid():
                    os.kill(pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
            os.remove(self.pid_file)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pid_read_write(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        daemon._write_pid(12345)
        self.assertEqual(daemon._read_pid(), 12345)

    def test_pid_remove(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        daemon._write_pid(12345)
        daemon._remove_pid()
        self.assertIsNone(daemon._read_pid())

    def test_is_process_alive_self(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        self.assertTrue(daemon._is_process_alive(os.getpid()))

    def test_is_process_alive_fake(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        self.assertFalse(daemon._is_process_alive(999999))

    def test_status_not_running(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        status = daemon.status()
        self.assertFalse(status['running'])
        self.assertIsNone(status['pid'])

    def test_status_running(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        daemon._write_pid(os.getpid())
        status = daemon.status()
        self.assertTrue(status['running'])
        self.assertEqual(status['pid'], os.getpid())

    def test_is_running_false(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        self.assertFalse(daemon.is_running())

    def test_is_running_true(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        daemon._write_pid(os.getpid())
        self.assertTrue(daemon.is_running())

    def test_is_running_stale_pid(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        daemon._write_pid(999999)
        self.assertFalse(daemon.is_running())

    def test_stop_not_running(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        self.assertFalse(daemon.stop())

    def test_register_worker(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        def worker():
            pass
        daemon.register_worker(worker)
        self.assertEqual(daemon._worker, worker)

    def test_register_shutdown_hook(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        def hook():
            pass
        daemon.register_shutdown_hook(hook)
        self.assertIn(hook, daemon._shutdown_hooks)

    def test_run_worker(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        called = []

        def worker():
            called.append(1)
            daemon._running = False  # 立即退出

        daemon.register_worker(worker)
        daemon._run()
        self.assertEqual(len(called), 1)

    def test_shutdown_hooks_called(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        hooks_called = []

        def hook1():
            hooks_called.append(1)

        def hook2():
            hooks_called.append(2)

        daemon.register_shutdown_hook(hook1)
        daemon.register_shutdown_hook(hook2)
        daemon._shutdown()

        self.assertEqual(len(hooks_called), 2)

    def test_signal_handler(self):
        daemon = DaemonManager(pid_file=self.pid_file)
        daemon._running = True
        daemon._signal_handler(signal.SIGTERM, None)
        self.assertFalse(daemon._running)


class TestRunDaemonCommand(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pid_file = os.path.join(self.temp_dir, 'test.pid')

    def tearDown(self):
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                if pid != os.getpid():
                    os.kill(pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
            os.remove(self.pid_file)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_status_not_running(self):
        def worker():
            pass
        result = run_daemon_command('status', worker, self.pid_file)
        self.assertEqual(result, 0)

    def test_stop_not_running(self):
        def worker():
            pass
        result = run_daemon_command('stop', worker, self.pid_file)
        self.assertEqual(result, 1)

    def test_unknown_command(self):
        def worker():
            pass
        result = run_daemon_command('unknown', worker, self.pid_file)
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
