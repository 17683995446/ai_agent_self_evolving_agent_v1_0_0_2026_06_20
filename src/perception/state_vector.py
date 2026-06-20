import abc
import logging
import os
import time
import psutil
from typing import Dict, Any, List, Optional

from .system_sensor import SystemSensor
from .log_sensor import LogSensor
from .file_sensor import FileSensor

logger = logging.getLogger(__name__)


class BaseStateVector(abc.ABC):
    """Abstract base class for state vector perception."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the base state vector.

        :param config: Configuration dictionary.
        :type config: Optional[Dict[str, Any]]
        :return: None
        :rtype: None
        """
        self.config = config or {}
        self.system_sensor = SystemSensor()
        self.log_sensor = LogSensor()
        self.file_sensor = FileSensor()
        self._self_pid = os.getpid()
        self._history_window = self._read_history_window(self.config)
        self._recent_mae_history: List[float] = []
        self._recent_task_success: List[float] = []
        self._coverage_ratio = 0.5

    @staticmethod
    def _read_history_window(config: Dict[str, Any]) -> int:
        """从配置中读取 history_window，兼容顶层和 memory.history_window 两种路径。

        :param config: 配置字典。
        :type config: Dict[str, Any]
        :return: history_window 值。
        :rtype: int
        """
        # 优先从 memory.history_window 读取（标准配置路径）
        memory_cfg = config.get('memory', {})
        if isinstance(memory_cfg, dict) and 'history_window' in memory_cfg:
            val = memory_cfg['history_window']
        elif 'history_window' in config:
            # 兼容顶层 history_window（旧格式）
            val = config['history_window']
        else:
            val = 100
        try:
            val = int(val)
            if val <= 0:
                val = 100
        except (TypeError, ValueError):
            val = 100
        return val

    @abc.abstractmethod
    def perceive(self) -> Dict[str, Any]:
        """Perceive full system state and return state vector.

        :param: 无
        :type: 无
        :return: State vector dictionary.
        :rtype: Dict[str, Any]
        """
        pass

    def _safe_perceive(self, sensor: Any) -> Dict[str, Any]:
        """Single sensor fault degradation: return empty data.

        :param sensor: Sensor instance.
        :type sensor: Any
        :return: Perception data or empty dictionary.
        :rtype: Dict[str, Any]
        """
        try:
            return sensor.perceive()
        except Exception as e:
            logger.debug("StateVector._safe_perceive: sensor %s failed: %s", type(sensor).__name__, e)
            return {}

    def _get_root_disk_percent(self, disk_data: Dict[str, Any]) -> float:
        """Return root partition disk usage percent.

        :param disk_data: Disk data dictionary.
        :type disk_data: Dict[str, Any]
        :return: Root disk usage percent.
        :rtype: float
        """
        partitions = disk_data.get('partitions', {})
        for mount, info in partitions.items():
            if mount == '/':
                return info.get('percent', 0.0)
        return self._get_first_partition_percent(partitions)

    def _get_first_partition_percent(self, partitions: Dict[str, Any]) -> float:
        """Return first partition's usage percent.

        :param partitions: Partitions dictionary.
        :type partitions: Dict[str, Any]
        :return: First partition usage percent.
        :rtype: float
        """
        if partitions:
            first = next(iter(partitions.values()))
            return first.get('percent', 0.0)
        return 0.0

    def _get_tmp_size(self, dir_stats: Dict[str, Any]) -> float:
        """Return tmp directory size in GB.

        :param dir_stats: Directory statistics dictionary.
        :type dir_stats: Dict[str, Any]
        :return: Tmp directory size in GB.
        :rtype: float
        """
        for path, stats in dir_stats.items():
            if path in ['/tmp', '/var/tmp']:
                return stats.get('total_size_mb', 0.0) / 1024.0
        return 0.0

    def _read_self(self) -> Dict[str, Any]:
        """Read self process metrics.

        :param: 无
        :type: 无
        :return: Self process metrics dictionary.
        :rtype: Dict[str, Any]
        """
        try:
            proc = psutil.Process(self._self_pid)
            return self._extract_proc_metrics(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {'cpu_percent': 0.0, 'mem_mb': 0.0}

    def _extract_proc_metrics(self, proc: Any) -> Dict[str, Any]:
        """Extract metrics from a process object.

        :param proc: Process object.
        :type proc: Any
        :return: Process metrics dictionary.
        :rtype: Dict[str, Any]
        """
        cpu_percent = proc.cpu_percent(interval=None)
        mem_info = proc.memory_info()
        return {
            'cpu_percent': round(cpu_percent, 4),
            'mem_mb': round(mem_info.rss / (1024 ** 2), 2),
        }

    def _get_recent_mae(self) -> float:
        """Return the recent mean absolute error.

        :param: 无
        :type: 无
        :return: Recent mean absolute error.
        :rtype: float
        """
        if not self._recent_mae_history:
            return 0.5
        return round(sum(self._recent_mae_history) / len(self._recent_mae_history), 4)

    def _get_recent_success_rate(self) -> float:
        """Return the recent task success rate.

        :param: 无
        :type: 无
        :return: Recent task success rate.
        :rtype: float
        """
        if not self._recent_task_success:
            return 0.5
        return round(sum(self._recent_task_success) / len(self._recent_task_success), 4)

    def update_mae(self, mae: float) -> None:
        """Update the recent MAE history.

        :param mae: Mean absolute error value.
        :type mae: float
        :return: None
        :rtype: None
        """
        self._recent_mae_history.append(mae)
        if len(self._recent_mae_history) > self._history_window:
            self._recent_mae_history.pop(0)

    def update_task_success(self, success: bool) -> None:
        """Update the recent task success history.

        :param success: Whether the task succeeded.
        :type success: bool
        :return: None
        :rtype: None
        """
        self._recent_task_success.append(1.0 if success else 0.0)
        if len(self._recent_task_success) > self._history_window:
            self._recent_task_success.pop(0)

    def update_coverage(self, ratio: float) -> None:
        """Update the coverage ratio.

        :param ratio: Coverage ratio.
        :type ratio: float
        :return: None
        :rtype: None
        """
        self._coverage_ratio = max(0.0, min(1.0, ratio))

    def flatten(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested state into a single-level dictionary.

        :param state: State dictionary.
        :type state: Dict[str, Any]
        :return: Flattened dictionary.
        :rtype: Dict[str, float]
        """
        flat = {}
        for category, values in state.items():
            self._flatten_category(flat, category, values)
        return flat

    def _flatten_category(self, flat: Dict[str, float], category: str, values: Any) -> None:
        """Flatten a single category into the flat dictionary.

        :param flat: Flat dictionary.
        :type flat: Dict[str, float]
        :param category: Category name.
        :type category: str
        :param values: Category values.
        :type values: Any
        :return: None
        :rtype: None
        """
        if category == 'timestamp':
            return
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    flat[f'{category}.{key}'] = float(value)


class StateVector(BaseStateVector):
    """Concrete implementation of StateVector."""

    def perceive(self) -> Dict[str, Any]:
        """Perceive full system state and return state vector.

        :param: 无
        :type: 无
        :return: State vector dictionary.
        :rtype: Dict[str, Any]
        """
        system_data = self._safe_perceive(self.system_sensor)
        log_data = self._safe_perceive(self.log_sensor)
        file_data = self._safe_perceive(self.file_sensor)
        self_data = self._read_self()
        return {
            'timestamp': time.time(),
            'system': self._build_system_state(system_data),
            'logs': self._build_logs_state(log_data),
            'files': self._build_files_state(file_data),
            'self': self._build_self_state(self_data),
        }

    def _build_system_state(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build system state dictionary.

        :param system_data: System sensor data.
        :type system_data: Dict[str, Any]
        :return: System state dictionary.
        :rtype: Dict[str, Any]
        """
        cpu = system_data.get('cpu', {})
        load = system_data.get('load', {})
        memory = system_data.get('memory', {})
        disk = system_data.get('disk', {})
        network = system_data.get('network', {})
        processes = system_data.get('processes', {})
        return self._assemble_system_state(cpu, load, memory, disk, network, processes)

    def _assemble_system_state(self, cpu: Dict[str, Any], load: Dict[str, Any], memory: Dict[str, Any], disk: Dict[str, Any], network: Dict[str, Any], processes: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble system state from component data.

        :param cpu: CPU metrics.
        :type cpu: Dict[str, Any]
        :param load: Load averages.
        :type load: Dict[str, Any]
        :param memory: Memory metrics.
        :type memory: Dict[str, Any]
        :param disk: Disk metrics.
        :type disk: Dict[str, Any]
        :param network: Network metrics.
        :type network: Dict[str, Any]
        :param processes: Process metrics.
        :type processes: Dict[str, Any]
        :return: System state dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'cpu_percent': cpu.get('percent', 0.0),
            'cpu_load_1m': load.get('load1', 0.0),
            'cpu_load_5m': load.get('load5', 0.0),
            'mem_percent': memory.get('percent', 0.0),
            'mem_available_gb': memory.get('available_gb', 0.0),
            'disk_percent': self._get_root_disk_percent(disk),
            'disk_read_mb_s': disk.get('read_mb_s', 0.0),
            'disk_write_mb_s': disk.get('write_mb_s', 0.0),
            'net_sent_mb_s': network.get('sent_mb_s', 0.0),
            'net_recv_mb_s': network.get('recv_mb_s', 0.0),
            'process_count': processes.get('total_count', 0),
            'zombie_count': processes.get('zombie_count', 0),
        }

    def _build_logs_state(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build logs state dictionary.

        :param log_data: Log sensor data.
        :type log_data: Dict[str, Any]
        :return: Logs state dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'error_count_1m': log_data.get('error_count', 0),
            'warn_count_1m': log_data.get('warn_count', 0),
            'error_trend': log_data.get('error_trend', 0.0),
            'unique_error_types': log_data.get('unique_error_types', 0),
        }

    def _build_files_state(self, file_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build files state dictionary.

        :param file_data: File sensor data.
        :type file_data: Dict[str, Any]
        :return: Files state dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'tmp_size_gb': self._get_tmp_size(file_data.get('dir_stats', {})),
            'tmp_growth_rate': file_data.get('size_delta_mb', 0.0),
            'recent_changes_1m': file_data.get('recent_changes', 0),
            'large_files_count': file_data.get('large_files_count', 0),
        }

    def _build_self_state(self, self_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build self state dictionary.

        :param self_data: Self process metrics.
        :type self_data: Dict[str, Any]
        :return: Self state dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'self_cpu_percent': self_data.get('cpu_percent', 0.0),
            'self_mem_mb': self_data.get('mem_mb', 0.0),
            'recent_mae': self._get_recent_mae(),
            'recent_tasks_success_rate': self._get_recent_success_rate(),
            'coverage_ratio': self._coverage_ratio,
        }
