#!/usr/bin/env python3
"""
跨平台兼容性测试
目标：验证代码在不同环境下的兼容性，路径处理正确，信号处理安全
"""

import os
import sys
import unittest
import tempfile
import shutil
import platform
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config
from src.optimization.snapshot_manager import SnapshotManager


class TestPathCompatibility(unittest.TestCase):
    """路径兼容性测试"""

    def test_path_join_uses_os_sep(self):
        """路径拼接应使用当前系统的分隔符"""
        path = os.path.join('data', 'memory.db')
        if os.name == 'nt':
            self.assertIn('\\', path)
        else:
            self.assertIn('/', path)

    def test_snapshot_manager_creates_nested_dirs(self):
        """SnapshotManager应能创建嵌套目录"""
        temp_dir = tempfile.mkdtemp()
        nested = os.path.join(temp_dir, 'level1', 'level2')
        sm = SnapshotManager(snapshot_dir=nested, max_count=5)
        self.assertTrue(os.path.exists(nested))
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_config_ensure_paths(self):
        """ensure_paths应创建所有必要目录"""
        Config._instance = None
        Config._config = {}
        temp_dir = tempfile.mkdtemp()
        c = Config()
        c._config = {
            'paths': {
                'data_dir': os.path.join(temp_dir, 'data'),
                'snapshot_dir': os.path.join(temp_dir, 'snapshots'),
                'log_dir': os.path.join(temp_dir, 'logs'),
            }
        }
        c.ensure_paths()
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'data')))
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'snapshots')))
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'logs')))
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_absolute_vs_relative_paths(self):
        """绝对路径和相对路径都应正常工作"""
        # 绝对路径
        abs_dir = tempfile.mkdtemp()
        sm1 = SnapshotManager(snapshot_dir=abs_dir, max_count=2)
        self.assertTrue(os.path.exists(abs_dir))
        shutil.rmtree(abs_dir, ignore_errors=True)

        # 相对路径
        orig = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        sm2 = SnapshotManager(snapshot_dir='relative_snapshots', max_count=2)
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'relative_snapshots')))
        os.chdir(orig)
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestPythonVersionCompatibility(unittest.TestCase):
    """Python版本兼容性测试"""

    def test_python_version_minimum(self):
        """Python版本应>=3.8"""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        self.assertGreaterEqual(version.minor, 8)

    def test_type_hints_supported(self):
        """类型注解应被正确解析"""
        from src.prediction.base_predictor import ExponentialSmoother
        s = ExponentialSmoother(alpha=0.3)
        # 如果类型注解有问题，import就会失败
        self.assertIsNotNone(s)

    def test_f_strings_supported(self):
        """f-string应正常工作"""
        name = "test"
        value = 42
        result = f"{name}={value}"
        self.assertEqual(result, "test=42")

    def test_dict_merge_operator(self):
        """字典合并运算符|应可用（Python 3.9+）"""
        if sys.version_info >= (3, 9):
            d1 = {'a': 1}
            d2 = {'b': 2}
            merged = d1 | d2
            self.assertEqual(merged, {'a': 1, 'b': 2})

    def test_dataclasses_available(self):
        """dataclasses模块应可用"""
        import dataclasses
        self.assertIsNotNone(dataclasses)


class TestPsutilContainerCompatibility(unittest.TestCase):
    """psutil容器环境兼容性测试"""

    def test_psutil_import_success(self):
        """psutil应能成功导入"""
        import psutil
        self.assertIsNotNone(psutil)

    def test_cpu_percent_non_blocking(self):
        """cpu_percent非阻塞模式不应崩溃"""
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        self.assertIsInstance(cpu, float)
        self.assertGreaterEqual(cpu, 0.0)

    def test_virtual_memory_available(self):
        """virtual_memory应可用"""
        import psutil
        mem = psutil.virtual_memory()
        self.assertIsNotNone(mem)
        self.assertGreater(mem.total, 0)

    def test_process_iter_with_attrs(self):
        """process_iter应支持attrs参数"""
        import psutil
        count = 0
        for proc in psutil.process_iter(['pid', 'name']):
            self.assertIsNotNone(proc.info)
            count += 1
            if count >= 5:
                break
        self.assertGreater(count, 0)

    def test_disk_usage_fallback(self):
        """disk_usage在受限环境应优雅降级"""
        import psutil
        try:
            usage = psutil.disk_usage('/')
            self.assertGreater(usage.total, 0)
        except PermissionError:
            # 容器环境可能权限受限
            pass

    def test_net_io_counters_none_handling(self):
        """net_io_counters可能返回None"""
        import psutil
        counters = psutil.net_io_counters(pernic=True)
        # 即使为None也不应崩溃
        if counters is not None:
            self.assertIsInstance(counters, dict)


class TestSignalHandling(unittest.TestCase):
    """信号处理测试"""

    def test_sigint_handler_settable(self):
        """SIGINT处理程序应可设置"""
        original = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            pass

        signal.signal(signal.SIGINT, handler)
        self.assertEqual(signal.getsignal(signal.SIGINT), handler)
        signal.signal(signal.SIGINT, original)

    def test_sigterm_handler_settable(self):
        """SIGTERM处理程序应可设置"""
        original = signal.getsignal(signal.SIGTERM)

        def handler(signum, frame):
            pass

        signal.signal(signal.SIGTERM, handler)
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)
        signal.signal(signal.SIGTERM, original)


class TestEncodingCompatibility(unittest.TestCase):
    """编码兼容性测试"""

    def test_utf8_file_operations(self):
        """UTF-8文件操作应正常"""
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, 'unicode.txt')
        content = '中文测试 UTF-8 \u4e2d\u6587'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        with open(path, 'r', encoding='utf-8') as f:
            read = f.read()
        self.assertEqual(read, content)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_json_unicode_roundtrip(self):
        """JSON Unicode往返应正常"""
        import json
        data = {'name': '中文', 'emoji': '\ud83d\ude80'}
        dumped = json.dumps(data, ensure_ascii=False)
        loaded = json.loads(dumped)
        self.assertEqual(loaded['name'], '中文')


class TestOSNameCompatibility(unittest.TestCase):
    """操作系统名称兼容性测试"""

    def test_os_name_recognized(self):
        """os.name应在预期值中"""
        self.assertIn(os.name, ['posix', 'nt', 'java'])

    def test_platform_info_available(self):
        """platform模块应能提供系统信息"""
        self.assertIsInstance(platform.system(), str)
        self.assertIsInstance(platform.release(), str)

    def test_tempfile_works(self):
        """tempfile应能创建临时文件"""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.assertTrue(os.path.exists(path))
        os.remove(path)


if __name__ == '__main__':
    unittest.main()
