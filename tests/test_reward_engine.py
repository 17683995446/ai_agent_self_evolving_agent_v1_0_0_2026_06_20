import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.drive.reward_engine import RewardEngine


class TestRewardEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RewardEngine(base_weights=(0.4, 0.4, 0.2))

    def _make_state(self, cpu=0.5, mem=0.5, disk=0.5, errors=0, zombies=0,
                    self_cpu=2.0, self_mem=128, coverage=0.5):
        return {
            'system': {
                'cpu_percent': cpu,
                'mem_percent': mem,
                'disk_percent': disk,
                'zombie_count': zombies,
            },
            'logs': {
                'error_count_1m': errors,
            },
            'self': {
                'self_cpu_percent': self_cpu,
                'self_mem_mb': self_mem,
                'coverage_ratio': coverage,
            }
        }

    def test_safe_reward_normal(self):
        state = self._make_state(cpu=0.5, mem=0.5, errors=0)
        reward = self.engine.compute_safe_reward(state)
        self.assertGreater(reward, 0.5)
        self.assertLessEqual(reward, 1.0)

    def test_safe_reward_high_cpu(self):
        state = self._make_state(cpu=95.0)
        reward = self.engine.compute_safe_reward(state)
        self.assertLess(reward, 0.7)

    def test_safe_reward_high_mem(self):
        state = self._make_state(mem=95.0)
        reward = self.engine.compute_safe_reward(state)
        self.assertLess(reward, 0.7)

    def test_safe_reward_high_disk(self):
        state = self._make_state(disk=97.0)
        reward = self.engine.compute_safe_reward(state)
        self.assertLess(reward, 0.85)

    def test_safe_reward_many_errors(self):
        state = self._make_state(errors=15)
        reward = self.engine.compute_safe_reward(state)
        self.assertLess(reward, 0.85)

    def test_safe_reward_zombies(self):
        state = self._make_state(zombies=10)
        reward = self.engine.compute_safe_reward(state)
        self.assertLess(reward, 1.0)

    def test_gain_reward_no_errors(self):
        state = self._make_state()
        reward = self.engine.compute_gain_reward(state, [0.1, 0.1, 0.1])
        self.assertGreater(reward, 0.5)

    def test_gain_reward_high_errors(self):
        state = self._make_state()
        reward = self.engine.compute_gain_reward(state, [0.8, 0.9, 0.7])
        self.assertLess(reward, 0.5)

    def test_gain_reward_empty_errors(self):
        state = self._make_state()
        reward = self.engine.compute_gain_reward(state, [])
        self.assertEqual(reward, 0.5)

    def test_gain_reward_coverage_boost(self):
        state_low = self._make_state(coverage=0.2)
        state_high = self._make_state(coverage=0.8)
        errors = [0.2, 0.2]
        reward_low = self.engine.compute_gain_reward(state_low, errors)
        reward_high = self.engine.compute_gain_reward(state_high, errors)
        self.assertLess(reward_low, reward_high)

    def test_eff_reward_low_usage(self):
        state = self._make_state(self_cpu=2.0, self_mem=64)
        reward = self.engine.compute_eff_reward(state)
        self.assertGreater(reward, 0.5)

    def test_eff_reward_high_usage(self):
        state = self._make_state(self_cpu=20.0, self_mem=512)
        reward = self.engine.compute_eff_reward(state)
        self.assertLess(reward, 0.5)

    def test_dynamic_weights_emergency(self):
        state = self._make_state(cpu=95.0)
        weights = self.engine.compute_dynamic_weights(state)
        self.assertEqual(weights[0], 0.8)

    def test_dynamic_weights_resource_tight(self):
        state = self._make_state(self_cpu=20.0)
        weights = self.engine.compute_dynamic_weights(state)
        self.assertEqual(weights[2], 0.8)

    def test_dynamic_weights_idle(self):
        state = self._make_state(cpu=10.0, errors=0)
        weights = self.engine.compute_dynamic_weights(state)
        self.assertEqual(weights[1], 0.7)

    def test_dynamic_weights_default(self):
        state = self._make_state(cpu=0.5, errors=3)
        weights = self.engine.compute_dynamic_weights(state)
        self.assertEqual(weights, (0.4, 0.4, 0.2))

    def test_total_reward_structure(self):
        state = self._make_state()
        result = self.engine.compute_total_reward(state, [0.2, 0.3])
        self.assertIn('total_reward', result)
        self.assertIn('safe_reward', result)
        self.assertIn('gain_reward', result)
        self.assertIn('eff_reward', result)
        self.assertIn('safe_gap', result)
        self.assertIn('gain_gap', result)
        self.assertIn('eff_gap', result)
        self.assertIn('weights', result)

    def test_gap_calculation(self):
        state = self._make_state()
        result = self.engine.compute_total_reward(state, [0.2, 0.3])
        self.assertAlmostEqual(
            result['safe_gap'],
            1.0 - result['safe_reward'],
            places=3
        )

    def test_total_reward_range(self):
        state = self._make_state()
        result = self.engine.compute_total_reward(state, [0.2, 0.3])
        self.assertGreaterEqual(result['total_reward'], 0.0)
        self.assertLessEqual(result['total_reward'], 1.0)

    def test_novelty_first_state(self):
        state = self._make_state()
        novelty = self.engine._compute_novelty(state)
        self.assertEqual(novelty, 1.0)

    def test_novelty_similar_state(self):
        state1 = self._make_state(cpu=0.5)
        state2 = self._make_state(cpu=0.51)
        self.engine._compute_novelty(state1)
        novelty = self.engine._compute_novelty(state2)
        self.assertLess(novelty, 1.0)
        self.assertGreaterEqual(novelty, 0.0)


if __name__ == '__main__':
    unittest.main()
