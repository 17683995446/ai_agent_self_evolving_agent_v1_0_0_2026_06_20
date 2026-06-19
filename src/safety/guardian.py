import abc
import logging
import os
import psutil
import time
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


class BaseResourceGuardian(abc.ABC):
    """Abstract base class for resource guardians."""

    def __init__(self, max_cpu_percent: float = 20.0,
                 max_mem_percent: float = 20.0,
                 max_disk_gb: float = 1.0,
                 data_dir: str = 'data') -> None:
        """Initialize the base resource guardian.

        :param max_cpu_percent: Maximum allowed CPU percent.
        :type max_cpu_percent: float
        :param max_mem_percent: Maximum allowed memory percent.
        :type max_mem_percent: float
        :param max_disk_gb: Maximum allowed disk usage in GB.
        :type max_disk_gb: float
        :param data_dir: Data directory path.
        :type data_dir: str
        :return: None
        :rtype: None
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_mem_percent = max_mem_percent
        self.max_disk_gb = max_disk_gb
        self.data_dir = data_dir
        self._self_pid = os.getpid()

    @abc.abstractmethod
    def _get_resource_metrics(self) -> Dict[str, float]:
        """Return current resource metrics.

        :param: 无
        :type: 无
        :return: Resource metrics dictionary.
        :rtype: Dict[str, float]
        """

    def check_resources(self) -> Dict[str, Any]:
        """Check resources and return result.

        :param: 无
        :type: 无
        :return: Resource check result dictionary.
        :rtype: Dict[str, Any]
        """
        metrics = self._get_resource_metrics()
        violations = self._check_violations(metrics)
        self._log_resource_check(metrics, violations)
        return self._build_resource_result(metrics, violations)

    def _log_resource_check(self, metrics: Dict[str, float],
                            violations: List[str]) -> None:
        """Log resource check results.

        :param metrics: Resource metrics dictionary.
        :type metrics: Dict[str, float]
        :param violations: List of violation messages.
        :type violations: List[str]
        :return: None
        :rtype: None
        """
        logger.debug("ResourceGuardian.check_resources: cpu=%.1f mem=%.1f safe=%s",
                     metrics['cpu_percent'], metrics['mem_percent'], len(violations) == 0)

    def _build_resource_result(self, metrics: Dict[str, float],
                               violations: List[str]) -> Dict[str, Any]:
        """Build resource check result dict.

        :param metrics: Resource metrics dictionary.
        :type metrics: Dict[str, float]
        :param violations: List of violation messages.
        :type violations: List[str]
        :return: Resource check result dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'cpu_percent': round(metrics['cpu_percent'], 2),
            'mem_percent': round(metrics['mem_percent'], 2),
            'mem_mb': round(metrics['mem_mb'], 2),
            'disk_mb': round(metrics['disk_mb'], 2),
            'violations': violations,
            'is_safe': len(violations) == 0,
        }

    def _check_violations(self, metrics: Dict[str, float]) -> List[str]:
        """Check violations.

        :param metrics: Resource metrics dictionary.
        :type metrics: Dict[str, float]
        :return: List of violation messages.
        :rtype: List[str]
        """
        violations = []
        self._check_cpu_violation(metrics, violations)
        self._check_mem_violation(metrics, violations)
        self._check_disk_violation(metrics, violations)
        return violations

    def _check_cpu_violation(self, metrics: Dict[str, float],
                             violations: List[str]) -> None:
        """Check CPU violation.

        :param metrics: Resource metrics dictionary.
        :type metrics: Dict[str, float]
        :param violations: List of violation messages.
        :type violations: List[str]
        :return: None
        :rtype: None
        """
        if metrics['cpu_percent'] > self.max_cpu_percent:
            violations.append(f"CPU {metrics['cpu_percent']:.1f}% > {self.max_cpu_percent}%")

    def _check_mem_violation(self, metrics: Dict[str, float],
                             violations: List[str]) -> None:
        """Check memory violation.

        :param metrics: Resource metrics dictionary.
        :type metrics: Dict[str, float]
        :param violations: List of violation messages.
        :type violations: List[str]
        :return: None
        :rtype: None
        """
        if metrics['mem_percent'] > self.max_mem_percent:
            violations.append(f"MEM {metrics['mem_percent']:.1f}% > {self.max_mem_percent}%")

    def _check_disk_violation(self, metrics: Dict[str, float],
                              violations: List[str]) -> None:
        """Check disk violation.

        :param metrics: Resource metrics dictionary.
        :type metrics: Dict[str, float]
        :param violations: List of violation messages.
        :type violations: List[str]
        :return: None
        :rtype: None
        """
        if metrics['disk_mb'] > self.max_disk_gb * 1024:
            violations.append(f"DISK {metrics['disk_mb']:.0f}MB > {self.max_disk_gb * 1024:.0f}MB")

    def _quick_dir_size(self, path: str, max_files: int = 50) -> float:
        """快速估算目录大小，限制扫描文件数

        :param path: Directory path.
        :type path: str
        :param max_files: Maximum files to scan.
        :type max_files: int
        :return: Directory size in MB.
        :rtype: float
        """
        if not os.path.exists(path):
            return 0.0
        return self._walk_dir_size(path, max_files)

    def _walk_dir_size(self, path: str, max_files: int) -> float:
        """Walk directory and accumulate file sizes up to max_files.

        :param path: Directory path.
        :type path: str
        :param max_files: Maximum files to scan.
        :type max_files: int
        :return: Directory size in MB.
        :rtype: float
        """
        total_mb = 0.0
        count = 0
        for root, dirs, files in os.walk(path):
            if count >= max_files:
                break
            total_mb, count = self._accumulate_files(root, files, total_mb, count, max_files)
        return total_mb

    def _accumulate_files(self, root: str, files: List[str],
                          total_mb: float, count: int, max_files: int) -> tuple:
        """Accumulate file sizes for a list of files.

        :param root: Root directory path.
        :type root: str
        :param files: List of file names.
        :type files: List[str]
        :param total_mb: Current total size in MB.
        :type total_mb: float
        :param count: Current file count.
        :type count: int
        :param max_files: Maximum files to scan.
        :type max_files: int
        :return: Updated total size and count.
        :rtype: tuple
        """
        for f in files:
            if count >= max_files:
                break
            total_mb, count = self._add_file_size(root, f, total_mb, count)
        return total_mb, count

    def _add_file_size(self, root: str, f: str, total_mb: float, count: int) -> tuple:
        """Add a single file size to total.

        :param root: Root directory path.
        :type root: str
        :param f: File name.
        :type f: str
        :param total_mb: Current total size in MB.
        :type total_mb: float
        :param count: Current file count.
        :type count: int
        :return: Updated total size and count.
        :rtype: tuple
        """
        try:
            total_mb += os.path.getsize(os.path.join(root, f)) / (1024 ** 2)
            count += 1
        except OSError:
            pass
        return total_mb, count


class ResourceGuardian(BaseResourceGuardian):
    """Concrete resource guardian implementation."""

    def _get_resource_metrics(self) -> Dict[str, float]:
        """Return current resource metrics.

        :param: 无
        :type: 无
        :return: Resource metrics dictionary.
        :rtype: Dict[str, float]
        """
        proc = psutil.Process(self._self_pid)
        cpu_percent = proc.cpu_percent(interval=None)
        mem_info = proc.memory_info()
        mem_mb = mem_info.rss / (1024 ** 2)
        mem_percent = self._get_mem_percent(mem_info)
        disk_mb = self._quick_dir_size(self.data_dir)
        return self._build_resource_metrics(cpu_percent, mem_percent, mem_mb, disk_mb)

    def _build_resource_metrics(self, cpu_percent: float, mem_percent: float, mem_mb: float, disk_mb: float) -> Dict[str, float]:
        """Build resource metrics dictionary.

        :param cpu_percent: CPU percent.
        :type cpu_percent: float
        :param mem_percent: Memory percent.
        :type mem_percent: float
        :param mem_mb: Memory usage in MB.
        :type mem_mb: float
        :param disk_mb: Disk usage in MB.
        :type disk_mb: float
        :return: Resource metrics dictionary.
        :rtype: Dict[str, float]
        """
        return {
            'cpu_percent': cpu_percent,
            'mem_percent': mem_percent,
            'mem_mb': mem_mb,
            'disk_mb': disk_mb,
        }

    def _get_mem_percent(self, mem_info: Any) -> float:
        """Calculate memory percent from mem_info.

        :param mem_info: Memory info object.
        :type mem_info: Any
        :return: Memory percent.
        :rtype: float
        """
        if not hasattr(self, '_total_mem'):
            self._total_mem = psutil.virtual_memory().total
        return (mem_info.rss / self._total_mem) * 100


class BaseBehaviorGuardian(abc.ABC):
    """Abstract base class for behavior guardians."""

    def __init__(self, forbidden_paths: Optional[List[str]] = None,
                 max_consecutive_failures: int = 5) -> None:
        """Initialize the base behavior guardian.

        :param forbidden_paths: List of forbidden paths.
        :type forbidden_paths: Optional[List[str]]
        :param max_consecutive_failures: Maximum consecutive failures allowed.
        :type max_consecutive_failures: int
        :return: None
        :rtype: None
        """
        self.forbidden_paths: Set[str] = self._build_forbidden_paths(forbidden_paths)
        self.max_consecutive_failures = max_consecutive_failures
        self._failure_counts: Dict[str, int] = {}

    def _build_forbidden_paths(self, forbidden_paths: Optional[List[str]]) -> Set[str]:
        """Build default or provided forbidden paths set.

        :param forbidden_paths: List of forbidden paths.
        :type forbidden_paths: Optional[List[str]]
        :return: Set of forbidden paths.
        :rtype: Set[str]
        """
        return set(forbidden_paths or [
            '/boot', '/etc', '/proc', '/sys', '/dev',
            '/bin', '/sbin', '/lib', '/lib64',
            '/usr/bin', '/usr/sbin',
        ])

    @abc.abstractmethod
    def validate_behavior(self, action: str) -> bool:
        """Validate a behavior action.

        :param action: Action string.
        :type action: str
        :return: Whether the action is valid.
        :rtype: bool
        """

    def check_path(self, path: str) -> bool:
        """Check if a path is allowed.

        :param path: Path to check.
        :type path: str
        :return: Whether the path is allowed.
        :rtype: bool
        """
        if path is None:
            return True
        abs_path = os.path.abspath(path)
        return self._is_path_allowed(abs_path)

    def _is_path_allowed(self, abs_path: str) -> bool:
        """Determine if absolute path is allowed.

        :param abs_path: Absolute path.
        :type abs_path: str
        :return: Whether the path is allowed.
        :rtype: bool
        """
        for forbidden in self.forbidden_paths:
            if abs_path.startswith(forbidden):
                logger.debug("BehaviorGuardian.check_path: forbidden path=%s", abs_path)
                return False
        logger.debug("BehaviorGuardian.check_path: allowed path=%s", abs_path)
        return True

    def check_exploration(self, task_id: str, success: bool) -> bool:
        """Check exploration result and update failure count.

        :param task_id: Task identifier.
        :type task_id: str
        :param success: Whether the exploration succeeded.
        :type success: bool
        :return: Whether the exploration is allowed to continue.
        :rtype: bool
        """
        if success:
            self._failure_counts[task_id] = 0
            return True
        self._failure_counts[task_id] = self._failure_counts.get(task_id, 0) + 1
        return self._failure_counts[task_id] <= self.max_consecutive_failures

    def is_frozen(self, task_id: str) -> bool:
        """Return whether task has reached max consecutive failures.

        :param task_id: Task identifier.
        :type task_id: str
        :return: Whether the task is frozen.
        :rtype: bool
        """
        return self._failure_counts.get(task_id, 0) >= self.max_consecutive_failures

    def get_failure_count(self, task_id: str) -> int:
        """Return current failure count for a task.

        :param task_id: Task identifier.
        :type task_id: str
        :return: Failure count.
        :rtype: int
        """
        return self._failure_counts.get(task_id, 0)

    def reset_failures(self, task_id: str) -> None:
        """Reset failure count for a task.

        :param task_id: Task identifier.
        :type task_id: str
        :return: None
        :rtype: None
        """
        self._failure_counts[task_id] = 0


class BehaviorGuardian(BaseBehaviorGuardian):
    """Concrete behavior guardian implementation."""

    def validate_behavior(self, action: str) -> bool:
        """Validate a behavior action.

        :param action: Action string.
        :type action: str
        :return: Whether the action is valid.
        :rtype: bool
        """
        return self.check_path(action)


class BaseSafetyEnforcer(abc.ABC):
    """Abstract base class for safety enforcers."""

    def __init__(self, resource_guardian: ResourceGuardian,
                 behavior_guardian: BehaviorGuardian) -> None:
        """Initialize the base safety enforcer.

        :param resource_guardian: Resource guardian instance.
        :type resource_guardian: ResourceGuardian
        :param behavior_guardian: Behavior guardian instance.
        :type behavior_guardian: BehaviorGuardian
        :return: None
        :rtype: None
        """
        self.resource = resource_guardian
        self.behavior = behavior_guardian
        self._emergency_mode: bool = False
        self._last_check: Optional[Dict[str, Any]] = None

    @abc.abstractmethod
    def enforce_policy(self, policy: str) -> bool:
        """Enforce a safety policy.

        :param policy: Policy string.
        :type policy: str
        :return: Whether the policy is enforced.
        :rtype: bool
        """

    def full_check(self) -> Dict[str, Any]:
        """Run a full safety check and return status.

        :param: 无
        :type: 无
        :return: Full check result dictionary.
        :rtype: Dict[str, Any]
        """
        resource_result = self.resource.check_resources()
        self._last_check = resource_result
        self._update_emergency_mode(resource_result)
        return self._build_full_check_result(resource_result)

    def _update_emergency_mode(self, resource_result: Dict[str, Any]) -> None:
        """Update emergency mode based on resource result.

        :param resource_result: Resource check result.
        :type resource_result: Dict[str, Any]
        :return: None
        :rtype: None
        """
        if not resource_result['is_safe']:
            self._emergency_mode = True
        else:
            self._emergency_mode = False

    def _build_full_check_result(self, resource_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build full check result dict.

        :param resource_result: Resource check result.
        :type resource_result: Dict[str, Any]
        :return: Full check result dictionary.
        :rtype: Dict[str, Any]
        """
        metrics = self._build_full_check_metrics(resource_result)
        return {
            'emergency_mode': self._emergency_mode,
            'resource_safe': resource_result['is_safe'],
            'violations': resource_result['violations'],
            'metrics': metrics,
        }

    def _build_full_check_metrics(self, resource_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build metrics sub-dict for full check result.

        :param resource_result: Resource check result.
        :type resource_result: Dict[str, Any]
        :return: Metrics dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'cpu': resource_result['cpu_percent'],
            'mem': resource_result['mem_percent'],
            'disk_mb': resource_result['disk_mb'],
        }

    def clear_emergency(self) -> None:
        """Clear emergency mode.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._emergency_mode = False

    def is_emergency(self) -> bool:
        """Return whether system is in emergency mode.

        :param: 无
        :type: 无
        :return: Whether in emergency mode.
        :rtype: bool
        """
        return self._emergency_mode


class SafetyEnforcer(BaseSafetyEnforcer):
    """Concrete safety enforcer implementation."""

    def enforce_policy(self, policy: str) -> bool:
        """Enforce a safety policy.

        :param policy: Policy string.
        :type policy: str
        :return: Whether the policy is enforced.
        :rtype: bool
        """
        result = self.full_check()
        return result['resource_safe']
