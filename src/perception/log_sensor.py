import abc
import os
import re
import time
import glob
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque

from src.core.constants import DEFAULT_LOG_PATHS


class BaseLogSensor(abc.ABC):
    """Abstract base class for log perception."""

    def __init__(self, log_paths: Optional[List[str]] = None, window_seconds: int = 60) -> None:
        """Initialize the base log sensor.

        :param log_paths: List of log file paths.
        :type log_paths: Optional[List[str]]
        :param window_seconds: Time window in seconds.
        :type window_seconds: int
        :return: None
        :rtype: None
        """
        self.window_seconds = window_seconds
        self.log_paths = log_paths or self._default_log_paths()
        self._file_positions: Dict[str, int] = {}
        self._error_history: deque = deque(maxlen=1000)
        self._warn_history: deque = deque(maxlen=1000)
        self._error_patterns: List[re.Pattern] = [
            re.compile(r'\bERROR\b', re.IGNORECASE),
            re.compile(r'\bFATAL\b', re.IGNORECASE),
            re.compile(r'\bCRITICAL\b', re.IGNORECASE),
            re.compile(r'\bEXCEPTION\b', re.IGNORECASE),
            re.compile(r'\bFAIL\b', re.IGNORECASE),
        ]
        self._warn_patterns: List[re.Pattern] = [
            re.compile(r'\bWARN\b', re.IGNORECASE),
            re.compile(r'\bWARNING\b', re.IGNORECASE),
        ]

    @abc.abstractmethod
    def perceive(self) -> Dict[str, Any]:
        """Perceive log state and return metrics.

        :param: 无
        :type: 无
        :return: Log metrics dictionary.
        :rtype: Dict[str, Any]
        """
        pass

    def _default_log_paths(self) -> List[str]:
        """Return default system log paths that exist.

        :param: 无
        :type: 无
        :return: List of existing default log paths.
        :rtype: List[str]
        """
        return [p for p in DEFAULT_LOG_PATHS if os.path.exists(p)]

    def _scan_file(
        self,
        log_path: str,
        now: float,
        max_lines: int,
    ) -> Tuple[int, int, Set[str], Set[str]]:
        """Scan a single log file for errors and warnings.

        :param log_path: Log file path.
        :type log_path: str
        :param now: Current timestamp.
        :type now: float
        :param max_lines: Maximum lines to scan.
        :type max_lines: int
        :return: Total errors, total warnings, new error types, new warn types.
        :rtype: Tuple[int, int, Set[str], Set[str]]
        """
        total_errors = 0
        total_warns = 0
        new_error_types: Set[str] = set()
        new_warn_types: Set[str] = set()
        try:
            lines = self._read_new_lines(log_path)
            for i, line in enumerate(lines):
                if i >= max_lines:
                    break
                total_errors, total_warns, new_error_types, new_warn_types = self._classify_line(
                    line, now, total_errors, total_warns, new_error_types, new_warn_types,
                )
        except (PermissionError, OSError):
            pass
        return total_errors, total_warns, new_error_types, new_warn_types

    def _classify_line(self, line: str, now: float, total_errors: int, total_warns: int, new_error_types: Set[str], new_warn_types: Set[str]) -> Tuple[int, int, Set[str], Set[str]]:
        """Classify a single log line as error, warning, or neither.

        :param line: Log line.
        :type line: str
        :param now: Current timestamp.
        :type now: float
        :param total_errors: Current total errors.
        :type total_errors: int
        :param total_warns: Current total warnings.
        :type total_warns: int
        :param new_error_types: Current new error types.
        :type new_error_types: Set[str]
        :param new_warn_types: Current new warn types.
        :type new_warn_types: Set[str]
        :return: Updated total errors, total warnings, new error types, new warn types.
        :rtype: Tuple[int, int, Set[str], Set[str]]
        """
        if any(p.search(line) for p in self._error_patterns):
            total_errors += 1
            self._error_history.append((now, line))
            error_type = self._extract_error_type(line)
            if error_type:
                new_error_types.add(error_type)
        elif any(p.search(line) for p in self._warn_patterns):
            total_warns += 1
            self._warn_history.append((now, line))
            warn_type = self._extract_warn_type(line)
            if warn_type:
                new_warn_types.add(warn_type)
        return total_errors, total_warns, new_error_types, new_warn_types

    def _read_new_lines(self, log_path: str) -> List[str]:
        """Read new lines from a log file since last read.

        :param log_path: Log file path.
        :type log_path: str
        :return: List of new lines.
        :rtype: List[str]
        """
        try:
            current_size = os.path.getsize(log_path)
        except (OSError, PermissionError):
            return []
        last_position = self._file_positions.get(log_path, 0)
        if current_size < last_position:
            last_position = 0
        if current_size == last_position:
            return []
        return self._read_lines_from_position(log_path, last_position)

    def _read_lines_from_position(self, log_path: str, position: int) -> List[str]:
        """Read lines from a file starting at the given position.

        :param log_path: Log file path.
        :type log_path: str
        :param position: File position.
        :type position: int
        :return: List of lines.
        :rtype: List[str]
        """
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(position)
                lines = f.readlines()
                self._file_positions[log_path] = f.tell()
            return [line.strip() for line in lines if line.strip()]
        except (OSError, PermissionError):
            return []

    def _extract_error_type(self, line: str) -> Optional[str]:
        """Extract the error type from a log line.

        :param line: Log line.
        :type line: str
        :return: Error type or None.
        :rtype: Optional[str]
        """
        parts = line.split()
        for i, part in enumerate(parts):
            if any(p.search(part) for p in self._error_patterns):
                if i + 1 < len(parts):
                    return parts[i + 1].strip(':[]')
        return None

    def _extract_warn_type(self, line: str) -> Optional[str]:
        """Extract the warning type from a log line.

        :param line: Log line.
        :type line: str
        :return: Warning type or None.
        :rtype: Optional[str]
        """
        parts = line.split()
        for i, part in enumerate(parts):
            if any(p.search(part) for p in self._warn_patterns):
                if i + 1 < len(parts):
                    return parts[i + 1].strip(':[]')
        return None

    def _compute_error_trend(self, window_start: float) -> float:
        """Compute error trend within the time window.

        :param window_start: Window start timestamp.
        :type window_start: float
        :return: Error trend value.
        :rtype: float
        """
        recent_errors = [t for t, _ in self._error_history if t >= window_start]
        if len(recent_errors) < 2:
            return 0.0
        half_window = window_start + (time.time() - window_start) / 2
        first_half = sum(1 for t in recent_errors if t < half_window)
        second_half = sum(1 for t in recent_errors if t >= half_window)
        return self._calc_trend_value(first_half, second_half)

    def _calc_trend_value(self, first_half: int, second_half: int) -> float:
        """Calculate trend value from first and second half counts.

        :param first_half: First half count.
        :type first_half: int
        :param second_half: Second half count.
        :type second_half: int
        :return: Trend value.
        :rtype: float
        """
        if first_half == 0:
            return 1.0 if second_half > 0 else 0.0
        trend = (second_half - first_half) / first_half
        return round(max(-1.0, min(1.0, trend)), 4)

    def add_log_path(self, path: str) -> None:
        """Add a log file path to monitor.

        :param path: Log file path.
        :type path: str
        :return: None
        :rtype: None
        """
        if os.path.exists(path) and path not in self.log_paths:
            self.log_paths.append(path)

    def remove_log_path(self, path: str) -> None:
        """Remove a log file path from monitoring.

        :param path: Log file path.
        :type path: str
        :return: None
        :rtype: None
        """
        if path in self.log_paths:
            self.log_paths.remove(path)
        self._file_positions.pop(path, None)


class LogSensor(BaseLogSensor):
    """Concrete implementation of LogSensor."""

    def perceive(self) -> Dict[str, Any]:
        """Perceive log state and return metrics.

        :param: 无
        :type: 无
        :return: Log metrics dictionary.
        :rtype: Dict[str, Any]
        """
        now = time.time()
        window_start = now - self.window_seconds
        total_errors, total_warns, new_error_types, new_warn_types = self._scan_all_files(now)
        error_trend = self._compute_error_trend(window_start)
        return self._build_log_result(now, total_errors, total_warns, new_error_types, new_warn_types, error_trend)

    def _scan_all_files(self, now: float) -> Tuple[int, int, Set[str], Set[str]]:
        """Scan all log files and aggregate counts.

        :param now: Current timestamp.
        :type now: float
        :return: Total errors, total warnings, new error types, new warn types.
        :rtype: Tuple[int, int, Set[str], Set[str]]
        """
        total_errors = 0
        total_warns = 0
        new_error_types: Set[str] = set()
        new_warn_types: Set[str] = set()
        max_lines_per_file = 500
        for log_path in self.log_paths:
            if not os.path.exists(log_path):
                continue
            errors, warns, err_types, warn_types = self._scan_file(log_path, now, max_lines_per_file)
            total_errors += errors
            total_warns += warns
            new_error_types.update(err_types)
            new_warn_types.update(warn_types)
        return total_errors, total_warns, new_error_types, new_warn_types

    def _build_log_result(self, now: float, total_errors: int, total_warns: int, new_error_types: Set[str], new_warn_types: Set[str], error_trend: float) -> Dict[str, Any]:
        """Build the log perception result dictionary.

        :param now: Current timestamp.
        :type now: float
        :param total_errors: Total error count.
        :type total_errors: int
        :param total_warns: Total warning count.
        :type total_warns: int
        :param new_error_types: New error types.
        :type new_error_types: Set[str]
        :param new_warn_types: New warn types.
        :type new_warn_types: Set[str]
        :param error_trend: Error trend value.
        :type error_trend: float
        :return: Log perception result dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            'timestamp': now,
            'error_count': total_errors,
            'warn_count': total_warns,
            'unique_error_types': len(new_error_types),
            'unique_warn_types': len(new_warn_types),
            'error_trend': error_trend,
            'monitored_files': len(self.log_paths),
        }
