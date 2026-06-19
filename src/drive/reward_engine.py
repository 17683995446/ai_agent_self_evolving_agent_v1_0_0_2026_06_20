import abc
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

from src.core.constants import (
    CPU_HIGH_RISK, CPU_MED_RISK,
    MEM_HIGH_RISK, MEM_MED_RISK,
    DISK_HIGH_RISK,
    ERROR_HIGH_RATE, ERROR_MED_RATE,
    TARGET_SELF_CPU, TARGET_SELF_MEM_MB,
)

logger = logging.getLogger(__name__)


class BaseRewardEngine(abc.ABC):
    """Abstract base class for computing rewards."""

    def __init__(self, base_weights: tuple = (0.4, 0.4, 0.2)) -> None:
        self.base_weights = base_weights
        self._state_history: deque = deque(maxlen=100)

    @abc.abstractmethod
    def compute_safe_reward(self, state: Dict[str, Any]) -> float:
        """Compute the safety reward from system state."""
        pass

    def _cpu_risk(self, system: Dict[str, Any]) -> float:
        """Return CPU risk contribution."""
        cpu = system.get('cpu_percent', 0.0)
        if cpu > CPU_HIGH_RISK:
            return 0.3
        elif cpu > CPU_MED_RISK:
            return 0.15
        return 0.0

    def _mem_risk(self, system: Dict[str, Any]) -> float:
        """Return memory risk contribution."""
        mem = system.get('mem_percent', 0.0)
        if mem > MEM_HIGH_RISK:
            return 0.3
        elif mem > MEM_MED_RISK:
            return 0.15
        return 0.0

    def _disk_risk(self, system: Dict[str, Any]) -> float:
        """Return disk risk contribution."""
        disk = system.get('disk_percent', 0.0)
        return 0.2 if disk > DISK_HIGH_RISK else 0.0

    def _error_risk(self, logs: Dict[str, Any]) -> float:
        """Return error rate risk contribution."""
        errors = logs.get('error_count_1m', 0)
        if errors > ERROR_HIGH_RATE:
            return 0.2
        elif errors > ERROR_MED_RATE:
            return 0.1
        return 0.0

    def _zombie_risk(self, system: Dict[str, Any]) -> float:
        """Return zombie process risk contribution."""
        zombies = system.get('zombie_count', 0)
        if zombies > 5:
            return 0.1
        elif zombies > 0:
            return 0.05
        return 0.0

    def compute_gain_reward(self, state: Dict[str, Any], recent_errors: List[float]) -> float:
        """Compute the gain reward from recent errors."""
        if len(recent_errors) == 0:
            return 0.5
        gain_reward = self._compute_base_gain(recent_errors)
        coverage = state.get('self', {}).get('coverage_ratio', 0.5)
        gain_reward *= (0.5 + 0.5 * coverage)
        return round(max(0.0, min(1.0, gain_reward)), 4)

    def _compute_base_gain(self, recent_errors: List[float]) -> float:
        mae = float(np.mean(recent_errors))
        normalized_error = min(mae, 1.0)
        return 1.0 - normalized_error

    def compute_eff_reward(self, state: Dict[str, Any]) -> float:
        """Compute the efficiency reward from self resource usage."""
        self_data = state.get('self', {})
        cpu_usage = self_data.get('self_cpu_percent', 0.0)
        mem_usage = self_data.get('self_mem_mb', 0.0)
        cpu_eff = TARGET_SELF_CPU / max(cpu_usage, 0.1)
        mem_eff = TARGET_SELF_MEM_MB / max(mem_usage, 32.0)
        eff_reward = min(1.0, (cpu_eff + mem_eff) / 2.0)
        return round(eff_reward, 4)

    def compute_dynamic_weights(
        self,
        state: Dict[str, Any],
        base_weights: Optional[tuple] = None,
    ) -> tuple:
        """Compute dynamic reward weights based on system state."""
        W_safe_base, W_gain_base, W_eff_base = base_weights or self.base_weights
        system = state.get('system', {})
        self_data = state.get('self', {})
        dynamic = self._resolve_dynamic_weights(system, self_data, state)
        if dynamic is not None:
            return dynamic
        return (W_safe_base, W_gain_base, W_eff_base)

    def _resolve_dynamic_weights(self, system: Dict[str, Any],
                                  self_data: Dict[str, Any],
                                  state: Dict[str, Any]) -> Optional[tuple]:
        if self._is_emergency(system):
            return (0.8, 0.1, 0.1)
        if self._is_resource_tight(self_data):
            return (0.1, 0.1, 0.8)
        if self._is_idle(system, state):
            return (0.1, 0.7, 0.2)
        return None

    def _is_emergency(self, system: Dict[str, Any]) -> bool:
        """Check if system is in emergency state."""
        return (
            system.get('cpu_percent', 0.0) > CPU_HIGH_RISK or
            system.get('mem_percent', 0.0) > MEM_HIGH_RISK
        )

    def _is_resource_tight(self, self_data: Dict[str, Any]) -> bool:
        """Check if self resource usage is tight."""
        return self_data.get('self_cpu_percent', 0.0) > 15.0

    def _is_idle(self, system: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Check if system is idle and error-free."""
        logs = state.get('logs', {})
        return (
            system.get('cpu_percent', 0.0) < 30.0 and
            logs.get('error_count_1m', 0) == 0
        )

    def compute_total_reward(
        self,
        state: Dict[str, Any],
        recent_errors: List[float],
        base_weights: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """Compute total reward and return detailed breakdown."""
        if state is None:
            state = {}
        R_safe = self.compute_safe_reward(state)
        R_gain = self.compute_gain_reward(state, recent_errors)
        R_eff = self.compute_eff_reward(state)
        W_safe, W_gain, W_eff = self.compute_dynamic_weights(state, base_weights)
        R_total = W_safe * R_safe + W_gain * R_gain + W_eff * R_eff
        logger.debug("RewardEngine.compute_total_reward: R_total=%.4f", R_total)
        return self._build_reward_result(R_total, R_safe, R_gain, R_eff, W_safe, W_gain, W_eff)

    def _build_reward_result(self, R_total: float, R_safe: float, R_gain: float,
                             R_eff: float, W_safe: float, W_gain: float, W_eff: float) -> Dict[str, Any]:
        return {
            'total_reward': round(R_total, 4),
            'safe_reward': R_safe,
            'gain_reward': R_gain,
            'eff_reward': R_eff,
            'safe_gap': round(1.0 - R_safe, 4),
            'gain_gap': round(1.0 - R_gain, 4),
            'eff_gap': round(1.0 - R_eff, 4),
            'weights': (W_safe, W_gain, W_eff),
        }

    def _compute_novelty(self, state: Dict[str, Any]) -> float:
        """Compute novelty score compared to historical states."""
        flat_state = self._flatten_state(state)
        if not self._state_history:
            self._state_history.append(flat_state)
            return 1.0
        avg_distance = self._compute_avg_distance(flat_state)
        self._state_history.append(flat_state)
        return min(1.0, avg_distance)

    def _compute_avg_distance(self, flat_state: Dict[str, float]) -> float:
        distances = [self._state_distance(flat_state, h) for h in self._state_history]
        return float(np.mean(distances)) if distances else 1.0

    def _flatten_state(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested state into a single-level dictionary."""
        flat = {}
        for category, values in state.items():
            if category == 'timestamp':
                continue
            if isinstance(values, dict):
                self._flatten_category(flat, category, values)
        return flat

    def _flatten_category(self, flat: Dict[str, float], category: str,
                          values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, (int, float)):
                flat[f'{category}.{key}'] = float(value)

    def _state_distance(self, s1: Dict[str, float], s2: Dict[str, float]) -> float:
        """Compute average absolute difference between two flat states."""
        keys = set(s1.keys()) | set(s2.keys())
        if not keys:
            return 0.0
        total_diff = sum(self._key_diff(s1, s2, key) for key in keys)
        return total_diff / len(keys)

    def _key_diff(self, s1: Dict[str, float], s2: Dict[str, float], key: str) -> float:
        v1 = s1.get(key, 0.0)
        v2 = s2.get(key, 0.0)
        return abs(v1 - v2)


class RewardEngine(BaseRewardEngine):
    """Concrete implementation of RewardEngine."""

    def compute_safe_reward(self, state: Dict[str, Any]) -> float:
        """Compute the safety reward from system state."""
        risk_score = self._compute_risk_score(state)
        novelty_score = self._compute_novelty(state)
        risk_score += novelty_score * 0.1
        safe_reward = max(0.0, 1.0 - risk_score)
        return round(safe_reward, 4)

    def _compute_risk_score(self, state: Dict[str, Any]) -> float:
        system = state.get('system', {})
        logs = state.get('logs', {})
        return (
            self._cpu_risk(system)
            + self._mem_risk(system)
            + self._disk_risk(system)
            + self._error_risk(logs)
            + self._zombie_risk(system)
        )
