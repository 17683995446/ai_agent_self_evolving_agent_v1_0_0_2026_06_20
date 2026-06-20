import abc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


class BaseSystemSensor(abc.ABC):
    """Abstract base class for system perception."""

    def __init__(self) -> None:
        """Initialize the base system sensor.

        :param: 无
        :type: 无
        :return: None
        :rtype: None
        """
        self._last_disk_io: Optional[Any] = None
        self._last_net_io: Optional[Dict[str, Any]] = None
        self._last_time: Optional[float] = None

    @abc.abstractmethod
    def perceive(self) -> Dict[str, Any]:
        """Perceive current system state.

        :param: 无
        :type: 无
        :return: System state dictionary.
        :rtype: Dict[str, Any]
        """
        pass

    def _read_cpu(self) -> Dict[str, Any]:
        """Read CPU metrics.

        :param: 无
        :type: 无
        :return: CPU metrics dictionary.
        :rtype: Dict[str, Any]
        """
        cpu_percent = psutil.cpu_percent(interval=None)
        percpu = psutil.cpu_percent(interval=None, percpu=True)
        return self._build_cpu_result(cpu_percent, percpu)

    def _build_cpu_result(self, cpu_percent: float, percpu: List[float]) -> Dict[str, Any]:
        """Build CPU metrics result dictionary.

        :param cpu_percent: Overall CPU percent.
        :type cpu_percent: float
        :param percpu: Per-CPU percents.
        :type percpu: List[float]
        :return: CPU metrics dictionary.
        :rtype: Dict[str, Any]
        """
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        return {
            'percent': round(cpu_percent / 100.0, 4),
            'percpu': [round(p / 100.0, 4) for p in percpu],
            'count': cpu_count,
            'freq_mhz': cpu_freq.current if cpu_freq else None,
        }

    def _read_memory(self) -> Dict[str, Any]:
        """Read memory metrics.

        :param: 无
        :type: 无
        :return: Memory metrics dictionary.
        :rtype: Dict[str, Any]
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return self._build_memory_result(mem, swap)

    def _build_memory_result(self, mem: Any, swap: Any) -> Dict[str, Any]:
        """Build memory metrics result dictionary.

        :param mem: Virtual memory info.
        :type mem: Any
        :param swap: Swap memory info.
        :type swap: Any
        :return: Memory metrics dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'total_gb': round(mem.total / (1024 ** 3), 2),
            'available_gb': round(mem.available / (1024 ** 3), 2),
            'percent': round(mem.percent / 100.0, 4),
            'used_gb': round(mem.used / (1024 ** 3), 2),
            'cached_gb': round(getattr(mem, 'cached', 0) / (1024 ** 3), 2),
            'buffers_gb': round(getattr(mem, 'buffers', 0) / (1024 ** 3), 2),
            'swap_percent': round(swap.percent / 100.0, 4),
            'swap_used_gb': round(swap.used / (1024 ** 3), 2),
        }

    def _read_disk(self) -> Dict[str, Any]:
        """Read disk metrics.

        :param: 无
        :type: 无
        :return: Disk metrics dictionary.
        :rtype: Dict[str, Any]
        """
        disk_info = self._collect_partition_usage()
        disk_io = psutil.disk_io_counters()
        io_rates = self._compute_io_rates(disk_io)
        return {
            'partitions': disk_info,
            'read_mb_s': io_rates.get('read_mb_s', 0.0),
            'write_mb_s': io_rates.get('write_mb_s', 0.0),
        }

    def _collect_partition_usage(self) -> Dict[str, Any]:
        """Collect partition usage information.

        :param: 无
        :type: 无
        :return: Partition usage dictionary.
        :rtype: Dict[str, Any]
        """
        partitions = psutil.disk_partitions()
        disk_info = {}
        for part in partitions:
            disk_info = self._add_partition_usage(part, disk_info)
        return disk_info

    def _add_partition_usage(self, part: Any, disk_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add a single partition's usage to disk_info.

        :param part: Partition object.
        :type part: Any
        :param disk_info: Existing disk info dictionary.
        :type disk_info: Dict[str, Any]
        :return: Updated disk info dictionary.
        :rtype: Dict[str, Any]
        """
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disk_info[part.mountpoint] = {
                'total_gb': round(usage.total / (1024 ** 3), 2),
                'used_gb': round(usage.used / (1024 ** 3), 2),
                'free_gb': round(usage.free / (1024 ** 3), 2),
                'percent': round(usage.percent / 100.0, 4),
            }
        except PermissionError as e:
            logger.debug("SystemSensor: disk partition %s permission denied: %s", part.mountpoint, e)
        return disk_info

    def _compute_io_rates(self, disk_io: Optional[Any]) -> Dict[str, float]:
        """Compute disk IO rates.

        :param disk_io: Disk IO counters.
        :type disk_io: Optional[Any]
        :return: IO rates dictionary.
        :rtype: Dict[str, float]
        """
        if disk_io is None:
            return {'read_mb_s': 0.0, 'write_mb_s': 0.0}
        now = time.time()
        if self._last_disk_io is None or self._last_time is None:
            self._last_disk_io = disk_io
            self._last_time = now
            return {'read_mb_s': 0.0, 'write_mb_s': 0.0}
        dt = now - self._last_time
        if dt <= 0:
            return {'read_mb_s': 0.0, 'write_mb_s': 0.0}
        return self._calc_disk_io_rates(disk_io, dt, now)

    def _calc_disk_io_rates(self, disk_io: Any, dt: float, now: float) -> Dict[str, float]:
        """Calculate disk IO rates and update state.

        :param disk_io: Disk IO counters.
        :type disk_io: Any
        :param dt: Time delta.
        :type dt: float
        :param now: Current timestamp.
        :type now: float
        :return: IO rates dictionary.
        :rtype: Dict[str, float]
        """
        read_mb_s = self._calc_rate(disk_io.read_bytes, self._last_disk_io.read_bytes, dt)
        write_mb_s = self._calc_rate(disk_io.write_bytes, self._last_disk_io.write_bytes, dt)
        self._last_disk_io = disk_io
        self._last_time = now
        return {
            'read_mb_s': round(max(0.0, read_mb_s), 4),
            'write_mb_s': round(max(0.0, write_mb_s), 4),
        }

    def _calc_rate(self, new_bytes: int, old_bytes: int, dt: float) -> float:
        """Calculate rate in MB/s.

        :param new_bytes: New byte count.
        :type new_bytes: int
        :param old_bytes: Old byte count.
        :type old_bytes: int
        :param dt: Time delta.
        :type dt: float
        :return: Rate in MB/s.
        :rtype: float
        """
        return (new_bytes - old_bytes) / (1024 ** 2) / dt

    def _read_network(self) -> Dict[str, Any]:
        """Read network metrics.

        :param: 无
        :type: 无
        :return: Network metrics dictionary.
        :rtype: Dict[str, Any]
        """
        net_io = psutil.net_io_counters(pernic=True)
        if net_io is None:
            return self._empty_network_result()
        total_sent, total_recv = self._aggregate_net_io(net_io)
        net_rates = self._compute_net_rates(total_sent, total_recv)
        connections = self._count_connections()
        return self._build_network_result(net_rates, connections)

    def _build_network_result(self, net_rates: Dict[str, float], connections: int) -> Dict[str, Any]:
        """Build network metrics result dictionary.

        :param net_rates: Network rates dictionary.
        :type net_rates: Dict[str, float]
        :param connections: Number of connections.
        :type connections: int
        :return: Network metrics dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'sent_mb_s': net_rates.get('sent_mb_s', 0.0),
            'recv_mb_s': net_rates.get('recv_mb_s', 0.0),
            'connections': connections,
        }

    def _empty_network_result(self) -> Dict[str, Any]:
        """Return empty network metrics result.

        :param: 无
        :type: 无
        :return: Empty network metrics dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'sent_mb_s': 0.0,
            'recv_mb_s': 0.0,
            'connections': 0,
        }

    def _aggregate_net_io(self, net_io: Dict[str, Any]) -> Tuple[int, int]:
        """Aggregate non-loopback network IO.

        :param net_io: Network IO counters per interface.
        :type net_io: Dict[str, Any]
        :return: Total sent and received bytes.
        :rtype: Tuple[int, int]
        """
        total_sent = 0
        total_recv = 0
        for nic, counters in net_io.items():
            if nic == 'lo':
                continue
            total_sent += counters.bytes_sent
            total_recv += counters.bytes_recv
        return total_sent, total_recv

    def _count_connections(self) -> int:
        """Count active network connections.

        :param: 无
        :type: 无
        :return: Number of connections.
        :rtype: int
        """
        try:
            return len(psutil.net_connections())
        except (psutil.AccessDenied, OSError):
            return 0

    def _compute_net_rates(self, sent: int, recv: int) -> Dict[str, float]:
        """Compute network transfer rates.

        :param sent: Total sent bytes.
        :type sent: int
        :param recv: Total received bytes.
        :type recv: int
        :return: Network rates dictionary.
        :rtype: Dict[str, float]
        """
        now = time.time()
        if self._last_net_io is None:
            self._last_net_io = {'sent': sent, 'recv': recv, 'time': now}
            return {'sent_mb_s': 0.0, 'recv_mb_s': 0.0}
        dt = now - self._last_net_io['time']
        if dt <= 0:
            return {'sent_mb_s': 0.0, 'recv_mb_s': 0.0}
        return self._calc_net_rates(sent, recv, dt, now)

    def _calc_net_rates(self, sent: int, recv: int, dt: float, now: float) -> Dict[str, float]:
        """Calculate network rates and update state.

        :param sent: Total sent bytes.
        :type sent: int
        :param recv: Total received bytes.
        :type recv: int
        :param dt: Time delta.
        :type dt: float
        :param now: Current timestamp.
        :type now: float
        :return: Network rates dictionary.
        :rtype: Dict[str, float]
        """
        sent_mb_s = self._calc_rate(sent, self._last_net_io['sent'], dt)
        recv_mb_s = self._calc_rate(recv, self._last_net_io['recv'], dt)
        self._last_net_io = {'sent': sent, 'recv': recv, 'time': now}
        return {
            'sent_mb_s': round(max(0.0, sent_mb_s), 4),
            'recv_mb_s': round(max(0.0, recv_mb_s), 4),
        }

    def _read_processes(self) -> Dict[str, Any]:
        """Read process metrics.

        :param: 无
        :type: 无
        :return: Process metrics dictionary.
        :rtype: Dict[str, Any]
        """
        pids = psutil.pids()
        zombie_count, top_cpu, top_mem = self._collect_process_snapshot()
        return self._build_process_result(pids, zombie_count, top_cpu, top_mem)

    def _collect_process_snapshot(self) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Collect a snapshot of process metrics.

        :param: 无
        :type: 无
        :return: Zombie count, top CPU processes, top memory processes.
        :rtype: Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]
        """
        zombie_count = 0
        top_cpu: List[Dict[str, Any]] = []
        top_mem: List[Dict[str, Any]] = []
        max_procs = 100
        count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            if count >= max_procs:
                break
            zombie_count, top_cpu, top_mem, count = self._add_process_info(
                proc, zombie_count, top_cpu, top_mem, count,
            )
        return zombie_count, top_cpu, top_mem

    def _add_process_info(self, proc: Any, zombie_count: int, top_cpu: List[Dict[str, Any]], top_mem: List[Dict[str, Any]], count: int) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """Add a single process's info to the snapshot.

        :param proc: Process object.
        :type proc: Any
        :param zombie_count: Current zombie count.
        :type zombie_count: int
        :param top_cpu: Top CPU processes list.
        :type top_cpu: List[Dict[str, Any]]
        :param top_mem: Top memory processes list.
        :type top_mem: List[Dict[str, Any]]
        :param count: Current process count.
        :type count: int
        :return: Updated zombie count, top_cpu, top_mem, and count.
        :rtype: Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], int]
        """
        try:
            if proc.info['status'] == psutil.STATUS_ZOMBIE:
                zombie_count += 1
            top_cpu.append({
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'cpu_percent': proc.info['cpu_percent'] or 0.0,
            })
            top_mem.append({
                'pid': proc.info['pid'],
                'name': proc.info['name'],
                'mem_percent': proc.info['memory_percent'] or 0.0,
            })
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug("SystemSensor: process scan skipped: %s", e)
        return zombie_count, top_cpu, top_mem, count

    def _build_process_result(self, pids: List[int], zombie_count: int, top_cpu: List[Dict[str, Any]], top_mem: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build process metrics result dictionary.

        :param pids: List of process IDs.
        :type pids: List[int]
        :param zombie_count: Number of zombie processes.
        :type zombie_count: int
        :param top_cpu: Top CPU processes.
        :type top_cpu: List[Dict[str, Any]]
        :param top_mem: Top memory processes.
        :type top_mem: List[Dict[str, Any]]
        :return: Process metrics dictionary.
        :rtype: Dict[str, Any]
        """
        top_cpu.sort(key=lambda x: x['cpu_percent'], reverse=True)
        top_mem.sort(key=lambda x: x['mem_percent'], reverse=True)
        return {
            'total_count': len(pids),
            'zombie_count': zombie_count,
            'top_cpu': top_cpu[:5],
            'top_mem': top_mem[:5],
        }

    def _read_load(self) -> Dict[str, float]:
        """Read system load averages.

        :param: 无
        :type: 无
        :return: Load averages dictionary.
        :rtype: Dict[str, float]
        """
        try:
            load1, load5, load15 = psutil.getloadavg()
            cpu_count = psutil.cpu_count() or 1
            return self._build_load_result(load1, load5, load15, cpu_count)
        except AttributeError:
            return {'load1': 0.0, 'load5': 0.0, 'load15': 0.0, 'load1_per_cpu': 0.0}

    def _build_load_result(self, load1: float, load5: float, load15: float, cpu_count: int) -> Dict[str, float]:
        """Build load averages result dictionary.

        :param load1: 1-minute load average.
        :type load1: float
        :param load5: 5-minute load average.
        :type load5: float
        :param load15: 15-minute load average.
        :type load15: float
        :param cpu_count: Number of CPUs.
        :type cpu_count: int
        :return: Load averages dictionary.
        :rtype: Dict[str, float]
        """
        return {
            'load1': round(load1, 4),
            'load5': round(load5, 4),
            'load15': round(load15, 4),
            'load1_per_cpu': round(load1 / cpu_count, 4),
        }


class SystemSensor(BaseSystemSensor):
    """Concrete implementation of SystemSensor."""

    def perceive(self) -> Dict[str, Any]:
        """Perceive current system state.

        :param: 无
        :type: 无
        :return: System state dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'timestamp': time.time(),
            'cpu': self._read_cpu(),
            'memory': self._read_memory(),
            'disk': self._read_disk(),
            'network': self._read_network(),
            'processes': self._read_processes(),
            'load': self._read_load(),
        }
