import os
import sys
import unittest
import tempfile
import shutil
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.safety.guardian import ResourceGuardian, BehaviorGuardian, SafetyEnforcer


class TestResourceGuardianExtreme(unittest.TestCase):
    """资源护栏在极限参数下的行为验证。"""

    def test_zero_limits_trigger_violation(self):
        """CPU和内存限制设为0应始终触发违规。"""
        guardian = ResourceGuardian(
            max_cpu_percent=0.0,
            max_mem_percent=0.0,
            max_disk_gb=0.0
        )
        result = guardian.check_resources()
        self.assertFalse(result['is_safe'])
        self.assertGreaterEqual(len(result['violations']), 1)

    def test_extreme_high_limits_never_trigger(self):
        """CPU和内存限制设为100%应始终不触发违规。"""
        guardian = ResourceGuardian(
            max_cpu_percent=100.0,
            max_mem_percent=100.0,
            max_disk_gb=10000.0
        )
        result = guardian.check_resources()
        self.assertTrue(result['is_safe'])
        self.assertEqual(len(result['violations']), 0)

    def test_negative_limits_behavior(self):
        """负值限制应始终触发违规（边界测试）。"""
        guardian = ResourceGuardian(
            max_cpu_percent=-1.0,
            max_mem_percent=-1.0
        )
        result = guardian.check_resources()
        self.assertFalse(result['is_safe'])

    def test_disk_size_on_nonexistent_path(self):
        """数据目录不存在时应返回0磁盘占用。"""
        guardian = ResourceGuardian(
            max_cpu_percent=100.0,
            max_mem_percent=100.0,
            max_disk_gb=-0.1,
            data_dir='nonexistent_path_xyz'
        )
        size = guardian._quick_dir_size('nonexistent_path_xyz')
        self.assertEqual(size, 0.0)
        result = guardian.check_resources()
        self.assertFalse(result['is_safe'])

    def test_quick_dir_size_limits_scan(self):
        """快速目录扫描应限制文件数量，不无限遍历。"""
        temp_dir = tempfile.mkdtemp()
        try:
            for i in range(100):
                with open(os.path.join(temp_dir, f'file_{i}.txt'), 'w') as f:
                    f.write('x' * 1024)
            guardian = ResourceGuardian()
            size = guardian._quick_dir_size(temp_dir, max_files=10)
            self.assertGreater(size, 0.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_concurrent_resource_checks(self):
        """并发资源检查不应崩溃。"""
        guardian = ResourceGuardian(max_cpu_percent=100.0, max_mem_percent=100.0)
        results = []
        errors = []

        def check():
            try:
                results.append(guardian.check_resources())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 20)
        for r in results:
            self.assertIn('cpu_percent', r)


class TestBehaviorGuardianExtreme(unittest.TestCase):
    """行为护栏在极限攻击场景下的防御验证。"""

    def test_path_traversal_attack(self):
        """路径遍历攻击（../etc/passwd）应被拦截。"""
        guardian = BehaviorGuardian()
        self.assertFalse(guardian.check_path('/tmp/../etc/passwd'))

    def test_symlink_attack_simulation(self):
        """模拟符号链接指向敏感目录应被拦截。"""
        guardian = BehaviorGuardian()
        self.assertFalse(guardian.check_path('/etc'))
        self.assertFalse(guardian.check_path('/etc/'))

    def test_none_path_allowed(self):
        """None路径应被允许（无目标路径任务）。"""
        guardian = BehaviorGuardian()
        self.assertTrue(guardian.check_path(None))

    def test_empty_path_allowed(self):
        """空路径应被允许（转换为当前目录）。"""
        guardian = BehaviorGuardian()
        self.assertTrue(guardian.check_path(''))

    def test_unicode_path_handling(self):
        """Unicode路径不应导致崩溃。"""
        guardian = BehaviorGuardian()
        result = guardian.check_path('/tmp/测试路径')
        self.assertTrue(result)

    def test_very_long_path(self):
        """超长路径不应导致崩溃。"""
        guardian = BehaviorGuardian()
        long_path = '/tmp/' + 'a' * 1000
        result = guardian.check_path(long_path)
        self.assertTrue(result)

    def test_exact_boundary_failure_count(self):
        """刚好达到max_consecutive_failures边界时应允许，超过时应冻结。"""
        guardian = BehaviorGuardian(max_consecutive_failures=3)
        self.assertTrue(guardian.check_exploration('t1', success=False))
        self.assertTrue(guardian.check_exploration('t1', success=False))
        self.assertTrue(guardian.check_exploration('t1', success=False))
        self.assertFalse(guardian.check_exploration('t1', success=False))
        self.assertTrue(guardian.is_frozen('t1'))

    def test_success_resets_failure_count(self):
        """成功一次应重置失败计数。"""
        guardian = BehaviorGuardian(max_consecutive_failures=3)
        guardian.check_exploration('t1', success=False)
        guardian.check_exploration('t1', success=False)
        guardian.check_exploration('t1', success=True)
        guardian.check_exploration('t1', success=False)
        self.assertFalse(guardian.is_frozen('t1'))

    def test_concurrent_failure_tracking(self):
        """并发失败计数不应丢失更新。"""
        guardian = BehaviorGuardian(max_consecutive_failures=1000)

        def fail_task():
            for _ in range(10):
                guardian.check_exploration('t1', success=False)

        threads = [threading.Thread(target=fail_task) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(guardian.get_failure_count('t1'), 100)

    def test_multiple_tasks_isolated(self):
        """不同任务的失败计数应相互隔离。"""
        guardian = BehaviorGuardian(max_consecutive_failures=3)
        guardian.check_exploration('t1', success=False)
        guardian.check_exploration('t1', success=False)
        guardian.check_exploration('t2', success=False)
        self.assertFalse(guardian.is_frozen('t1'))
        self.assertFalse(guardian.is_frozen('t2'))


class TestSafetyEnforcerExtreme(unittest.TestCase):
    """安全执行器在极限场景下的稳定性验证。"""

    def test_rapid_toggle_emergency(self):
        """快速切换紧急模式状态应保持一致。"""
        rg = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)

        for _ in range(50):
            result = se.full_check()
            self.assertTrue(result['emergency_mode'])
            se.clear_emergency()
            self.assertFalse(se.is_emergency())

    def test_concurrent_full_check(self):
        """并发full_check调用不应崩溃。"""
        rg = ResourceGuardian(max_cpu_percent=100.0, max_mem_percent=100.0)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        errors = []

        def check():
            try:
                se.full_check()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    def test_emergency_persists_until_cleared(self):
        """紧急模式应在full_check后持续，直到显式清除。"""
        rg = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        se.full_check()
        self.assertTrue(se.is_emergency())
        se.clear_emergency()
        self.assertFalse(se.is_emergency())

    def test_metrics_format_consistency(self):
        """full_check返回的metrics格式应保持一致。"""
        rg = ResourceGuardian(max_cpu_percent=100.0, max_mem_percent=100.0)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        result = se.full_check()
        self.assertIn('cpu', result['metrics'])
        self.assertIn('mem', result['metrics'])
        self.assertIn('disk_mb', result['metrics'])
        self.assertIsInstance(result['metrics']['cpu'], float)
        self.assertIsInstance(result['metrics']['mem'], float)


class TestSafetyIntegrationExtreme(unittest.TestCase):
    """安全模块端到端极限集成测试。"""

    def test_full_safety_pipeline_under_attack(self):
        """模拟攻击场景：恶意路径+资源超限+连续失败。"""
        rg = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        bg = BehaviorGuardian(max_consecutive_failures=2)
        se = SafetyEnforcer(rg, bg)

        safety = se.full_check()
        self.assertTrue(safety['emergency_mode'])

        self.assertFalse(bg.check_path('/etc/passwd'))
        self.assertTrue(bg.check_path('/tmp/safe'))

        bg.check_exploration('attack_task', success=False)
        bg.check_exploration('attack_task', success=False)
        bg.check_exploration('attack_task', success=False)
        self.assertTrue(bg.is_frozen('attack_task'))

    def test_behavior_guardian_custom_forbidden(self):
        """自定义禁止路径列表应生效。"""
        guardian = BehaviorGuardian(forbidden_paths=['/secret', '/private'])
        self.assertFalse(guardian.check_path('/secret/data'))
        self.assertTrue(guardian.check_path('/etc/passwd'))


if __name__ == '__main__':
    unittest.main()
