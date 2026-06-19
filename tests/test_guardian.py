import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.safety.guardian import ResourceGuardian, BehaviorGuardian, SafetyEnforcer


class TestResourceGuardian(unittest.TestCase):
    def test_check_returns_dict(self):
        g = ResourceGuardian(max_cpu_percent=100.0, max_mem_percent=100.0)
        result = g.check_resources()
        self.assertIn('cpu_percent', result)
        self.assertIn('mem_percent', result)
        self.assertIn('is_safe', result)
        self.assertTrue(result['is_safe'])

    def test_violation_detection(self):
        g = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        result = g.check_resources()
        self.assertFalse(result['is_safe'])
        self.assertGreater(len(result['violations']), 0)


class TestBehaviorGuardian(unittest.TestCase):
    def test_forbidden_path(self):
        g = BehaviorGuardian()
        self.assertFalse(g.check_path('/etc/passwd'))
        self.assertFalse(g.check_path('/bin/ls'))

    def test_allowed_path(self):
        g = BehaviorGuardian()
        self.assertTrue(g.check_path('/tmp/test'))
        self.assertTrue(g.check_path('/workspace/test'))

    def test_exploration_success(self):
        g = BehaviorGuardian(max_consecutive_failures=3)
        self.assertTrue(g.check_exploration('t1', success=True))
        self.assertEqual(g.get_failure_count('t1'), 0)

    def test_exploration_failure_limit(self):
        g = BehaviorGuardian(max_consecutive_failures=3)
        g.check_exploration('t1', success=False)
        g.check_exploration('t1', success=False)
        self.assertTrue(g.check_exploration('t1', success=False))
        self.assertFalse(g.check_exploration('t1', success=False))
        self.assertTrue(g.is_frozen('t1'))

    def test_reset_failures(self):
        g = BehaviorGuardian(max_consecutive_failures=3)
        g.check_exploration('t1', success=False)
        g.check_exploration('t1', success=False)
        g.reset_failures('t1')
        self.assertEqual(g.get_failure_count('t1'), 0)
        self.assertFalse(g.is_frozen('t1'))


class TestSafetyEnforcer(unittest.TestCase):
    def test_full_check(self):
        rg = ResourceGuardian(max_cpu_percent=100.0, max_mem_percent=100.0)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        result = se.full_check()
        self.assertIn('emergency_mode', result)
        self.assertIn('resource_safe', result)
        self.assertFalse(result['emergency_mode'])

    def test_emergency_trigger(self):
        rg = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        result = se.full_check()
        self.assertTrue(result['emergency_mode'])
        self.assertTrue(se.is_emergency())

    def test_clear_emergency(self):
        rg = ResourceGuardian(max_cpu_percent=0.01, max_mem_percent=0.01)
        bg = BehaviorGuardian()
        se = SafetyEnforcer(rg, bg)
        se.full_check()
        se.clear_emergency()
        self.assertFalse(se.is_emergency())


if __name__ == '__main__':
    unittest.main()
