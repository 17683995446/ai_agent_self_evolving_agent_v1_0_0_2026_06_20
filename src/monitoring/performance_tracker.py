import time
from typing import Any, Dict, List, Optional


class PerformanceTracker:
    """轻量级性能追踪器：记录操作延迟并生成可观测性摘要。

    - 支持按 operation 聚合：count / mean / min / max / p95
    - 支持自定义标签（tag）记录上下文，如 worker_id、skill_name、step_name
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        operation: str,
        latency: float,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录一次操作耗时。"""
        self._records.append({
            'operation': operation,
            'latency': latency,
            'tags': tags or {},
            'timestamp': time.time(),
        })

    def time(self, operation: str, tags: Optional[Dict[str, Any]] = None):
        """上下文管理器，自动测量代码块耗时。"""
        return _TrackContext(self, operation, tags)

    def _percentile(self, values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * pct / 100.0
        f = int(k)
        c = min(f + 1, len(sorted_values) - 1)
        if f == c:
            return sorted_values[f]
        return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

    def summary(self) -> Dict[str, Any]:
        """按 operation 聚合延迟统计。"""
        grouped: Dict[str, List[float]] = {}
        for r in self._records:
            grouped.setdefault(r['operation'], []).append(r['latency'])

        result: Dict[str, Any] = {}
        for op, latencies in grouped.items():
            result[op] = {
                'count': len(latencies),
                'mean': sum(latencies) / len(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'p95': self._percentile(latencies, 95.0),
            }
        return result

    def metrics(self) -> List[Dict[str, Any]]:
        """返回原始记录列表，用于详细可观测性。"""
        return list(self._records)

    def reset(self) -> None:
        """清空记录。"""
        self._records.clear()


class _TrackContext:
    def __init__(self, tracker: PerformanceTracker, operation: str, tags: Optional[Dict[str, Any]]) -> None:
        self.tracker = tracker
        self.operation = operation
        self.tags = tags
        self.start: Optional[float] = None

    def __enter__(self) -> '_TrackContext':
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start is not None:
            self.tracker.record(self.operation, time.time() - self.start, tags=self.tags)
