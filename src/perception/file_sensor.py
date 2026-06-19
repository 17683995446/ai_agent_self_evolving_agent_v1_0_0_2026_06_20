import abc
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from src.core.constants import DEFAULT_WATCH_PATHS


class BaseFileSensor(abc.ABC):
    """Abstract base class for file system perception."""

    def __init__(self, watch_paths: Optional[List[str]] = None, window_seconds: int = 60,
                 max_files_per_dir: int = 100, min_scan_interval: float = 5.0,
                 max_scan_depth: int = 3) -> None:
        """Initialize the base file sensor.

        :param watch_paths: List of watch paths.
        :type watch_paths: Optional[List[str]]
        :param window_seconds: Time window in seconds.
        :type window_seconds: int
        :param max_files_per_dir: Maximum files per directory.
        :type max_files_per_dir: int
        :param min_scan_interval: Minimum seconds between scans.
        :type min_scan_interval: float
        :param max_scan_depth: Maximum directory recursion depth.
        :type max_scan_depth: int
        :return: None
        :rtype: None
        """
        self.window_seconds = window_seconds
        self.watch_paths = watch_paths or DEFAULT_WATCH_PATHS
        self.max_files_per_dir = max_files_per_dir
        self.min_scan_interval = min_scan_interval
        self.max_scan_depth = max_scan_depth
        self._file_snapshots: Dict[str, Dict[str, Any]] = {}
        self._change_history: deque = deque(maxlen=1000)
        self._last_scan_time: Optional[float] = None
        self._cached_result: Optional[Dict[str, Any]] = None

    @abc.abstractmethod
    def perceive(self) -> Dict[str, Any]:
        """Perceive file system state and return metrics.

        :param: 无
        :type: 无
        :return: File system metrics dictionary.
        :rtype: Dict[str, Any]
        """
        pass

    def _scan_watch_path(
        self,
        watch_path: str,
        now: float,
        current_snapshots: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, int, int, int, Dict[str, Any]]:
        """Scan a single watch path and return change counts plus dir stats.

        :param watch_path: Watch path.
        :type watch_path: str
        :param now: Current timestamp.
        :type now: float
        :param current_snapshots: Current snapshots dictionary.
        :type current_snapshots: Dict[str, Dict[str, Any]]
        :return: New files, modified files, total size delta, large files count, dir stats.
        :rtype: Tuple[int, int, int, int, Dict[str, Any]]
        """
        new_files = 0
        modified_files = 0
        total_size_delta = 0
        large_files_count = 0
        file_count = 0
        total_size = 0
        root_depth = watch_path.rstrip(os.sep).count(os.sep)
        for root, dirs, files in os.walk(watch_path):
            current_depth = root.rstrip(os.sep).count(os.sep) - root_depth
            if current_depth >= self.max_scan_depth:
                del dirs[:]
                continue
            if file_count >= self.max_files_per_dir:
                break
            for filename in files:
                if file_count >= self.max_files_per_dir:
                    break
                filepath = os.path.join(root, filename)
                result = self._scan_single_file(filepath, now, current_snapshots)
                if result:
                    new_files += result[0]
                    modified_files += result[1]
                    total_size_delta += result[2]
                    large_files_count += result[3]
                    file_count += 1
                    total_size += current_snapshots[filepath]['size']
        dir_stats = {
            'file_count': file_count,
            'total_size_mb': round(total_size / (1024 ** 2), 2),
        }
        return new_files, modified_files, total_size_delta, large_files_count, dir_stats

    def _scan_single_file(self, filepath: str, now: float, current_snapshots: Dict[str, Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
        """Scan a single file and return change info or None on error.

        :param filepath: File path.
        :type filepath: str
        :param now: Current timestamp.
        :type now: float
        :param current_snapshots: Current snapshots dictionary.
        :type current_snapshots: Dict[str, Dict[str, Any]]
        :return: Change info tuple or None.
        :rtype: Optional[Tuple[int, int, int, int]]
        """
        try:
            stat = os.stat(filepath)
            snapshot = {
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'inode': stat.st_ino,
            }
            current_snapshots[filepath] = snapshot
            large = 1 if stat.st_size > 100 * 1024 * 1024 else 0
            new_files, modified_files, size_delta = self._compare_snapshot(filepath, snapshot, now)
            return new_files, modified_files, size_delta, large
        except (OSError, PermissionError):
            return None

    def _compare_snapshot(self, filepath: str, snapshot: Dict[str, Any], now: float) -> Tuple[int, int, int]:
        """Compare file snapshot with previous scan and return change counts.

        :param filepath: File path.
        :type filepath: str
        :param snapshot: Current snapshot.
        :type snapshot: Dict[str, Any]
        :param now: Current timestamp.
        :type now: float
        :return: New files, modified files, size delta.
        :rtype: Tuple[int, int, int]
        """
        if filepath in self._file_snapshots:
            old = self._file_snapshots[filepath]
            if old['size'] != snapshot['size'] or old['mtime'] != snapshot['mtime']:
                self._change_history.append((now, 'modified', filepath))
                return 0, 1, snapshot['size'] - old['size']
            return 0, 0, 0
        self._change_history.append((now, 'created', filepath))
        return 1, 0, 0

    def _count_deleted(self, now: float, current_snapshots: Dict[str, Dict[str, Any]]) -> int:
        """Count deleted files since last scan.

        :param now: Current timestamp.
        :type now: float
        :param current_snapshots: Current snapshots dictionary.
        :type current_snapshots: Dict[str, Dict[str, Any]]
        :return: Number of deleted files.
        :rtype: int
        """
        deleted_files = 0
        for old_path in self._file_snapshots:
            if old_path not in current_snapshots:
                deleted_files += 1
                self._change_history.append((now, 'deleted', old_path))
        return deleted_files

    def _should_use_cache(self, now: float) -> bool:
        """Check if cached result can be reused.

        :param now: Current timestamp.
        :type now: float
        :return: True if cache is valid.
        :rtype: bool
        """
        if self._cached_result is None:
            return False
        if self._last_scan_time is None:
            return False
        return (now - self._last_scan_time) < self.min_scan_interval

    def add_watch_path(self, path: str) -> None:
        """Add a directory path to watch.

        :param path: Directory path.
        :type path: str
        :return: None
        :rtype: None
        """
        if os.path.exists(path) and path not in self.watch_paths:
            self.watch_paths.append(path)

    def remove_watch_path(self, path: str) -> None:
        """Remove a directory path from watching.

        :param path: Directory path.
        :type path: str
        :return: None
        :rtype: None
        """
        if path in self.watch_paths:
            self.watch_paths.remove(path)
        keys_to_remove = [k for k in self._file_snapshots if k.startswith(path)]
        for k in keys_to_remove:
            del self._file_snapshots[k]


class FileSensor(BaseFileSensor):
    """Concrete implementation of FileSensor."""

    def perceive(self) -> Dict[str, Any]:
        """Perceive file system state and return metrics.

        :param: 无
        :type: 无
        :return: File system metrics dictionary.
        :rtype: Dict[str, Any]
        """
        now = time.time()
        if self._should_use_cache(now):
            return self._cached_result
        window_start = now - self.window_seconds
        total_new, total_modified, total_size_delta, total_large, current_snapshots, dir_stats = self._scan_all_watch_paths(now)
        deleted_files = self._count_deleted(now, current_snapshots)
        total_changes = total_new + deleted_files + total_modified
        recent_changes = sum(1 for t, _, _ in self._change_history if t >= window_start)
        self._file_snapshots = current_snapshots
        self._last_scan_time = now
        result = self._build_file_result(now, total_changes, total_new, deleted_files, total_modified, recent_changes, total_size_delta, total_large, current_snapshots, dir_stats)
        self._cached_result = result
        return result

    def _scan_all_watch_paths(self, now: float) -> Tuple[int, int, int, int, Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Scan all watch paths and return aggregated counts plus dir stats.

        :param now: Current timestamp.
        :type now: float
        :return: New files, modified files, size delta, large files count, current snapshots, dir stats.
        :rtype: Tuple[int, int, int, int, Dict[str, Dict[str, Any]], Dict[str, Any]]
        """
        total_new = 0
        total_modified = 0
        total_size_delta = 0
        total_large = 0
        current_snapshots: Dict[str, Dict[str, Any]] = {}
        dir_stats: Dict[str, Any] = {}
        for watch_path in self.watch_paths:
            if not os.path.exists(watch_path):
                continue
            new_files, modified_files, size_delta, large_count, stats = self._scan_watch_path(
                watch_path, now, current_snapshots,
            )
            total_new += new_files
            total_modified += modified_files
            total_size_delta += size_delta
            total_large += large_count
            dir_stats[watch_path] = stats
        return total_new, total_modified, total_size_delta, total_large, current_snapshots, dir_stats

    def _build_file_result(self, now: float, total_changes: int, total_new: int, deleted_files: int, total_modified: int, recent_changes: int, total_size_delta: int, total_large: int, current_snapshots: Dict[str, Dict[str, Any]], dir_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Build the file perception result dictionary.

        :param now: Current timestamp.
        :type now: float
        :param total_changes: Total changes count.
        :type total_changes: int
        :param total_new: Total new files.
        :type total_new: int
        :param deleted_files: Total deleted files.
        :type deleted_files: int
        :param total_modified: Total modified files.
        :type total_modified: int
        :param recent_changes: Recent changes count.
        :type recent_changes: int
        :param total_size_delta: Total size delta.
        :type total_size_delta: int
        :param total_large: Total large files count.
        :type total_large: int
        :param current_snapshots: Current snapshots dictionary.
        :type current_snapshots: Dict[str, Dict[str, Any]]
        :param dir_stats: Directory statistics dictionary.
        :type dir_stats: Dict[str, Any]
        :return: File perception result dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'timestamp': now,
            'total_changes': total_changes,
            'new_files': total_new,
            'deleted_files': deleted_files,
            'modified_files': total_modified,
            'recent_changes': recent_changes,
            'size_delta_mb': round(total_size_delta / (1024 ** 2), 4),
            'large_files_count': total_large,
            'monitored_dirs': len(self.watch_paths),
            'total_tracked_files': len(current_snapshots),
            'dir_stats': dir_stats,
        }
