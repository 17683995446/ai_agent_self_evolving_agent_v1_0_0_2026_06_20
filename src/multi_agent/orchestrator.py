import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from src.task.task_definition import Task

from .agent_worker import AgentWorker


class MultiAgentOrchestrator:
    """多 Agent 编排器：负责任务分发、能力匹配、结果聚合与任务自驱动调度。

    支持策略：
    - round_robin：轮询分配，适合同构 worker。
    - capability_match：按任务 required_capabilities 匹配 worker。
    - least_loaded：优先分配给当前任务数最少的 worker。

    :param strategy: 默认分配策略
    """

    def __init__(self, strategy: str = 'round_robin') -> None:
        if strategy not in self.supported_strategies():
            raise ValueError(f"Unsupported strategy: {strategy}")
        self._workers: Dict[str, AgentWorker] = {}
        self._strategy = strategy
        self._round_robin_index = 0
        self._missions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def supported_strategies() -> List[str]:
        """返回支持的分配策略。"""
        return ['round_robin', 'capability_match', 'least_loaded']

    def register_worker(self, worker: AgentWorker) -> None:
        """注册一个 AgentWorker。"""
        with self._lock:
            self._workers[worker.worker_id] = worker

    def unregister_worker(self, worker_id: str) -> bool:
        """注销指定 worker。"""
        with self._lock:
            return self._workers.pop(worker_id, None) is not None

    def list_workers(self) -> List[str]:
        """返回已注册 worker ID 列表。"""
        with self._lock:
            return list(self._workers.keys())

    def get_worker(self, worker_id: str) -> Optional[AgentWorker]:
        """根据 ID 获取 worker。"""
        with self._lock:
            return self._workers.get(worker_id)

    def set_strategy(self, strategy: str) -> None:
        """切换分配策略。"""
        if strategy not in self.supported_strategies():
            raise ValueError(f"Unsupported strategy: {strategy}")
        with self._lock:
            self._strategy = strategy

    def select_worker(self, task: Task) -> Optional[AgentWorker]:
        """按照当前策略为任务选择最合适的 worker。"""
        with self._lock:
            workers = list(self._workers.values())
            if not workers:
                return None

            if self._strategy == 'round_robin':
                return self._select_round_robin(workers)
            if self._strategy == 'capability_match':
                return self._select_capability_match(workers, task)
            if self._strategy == 'least_loaded':
                return self._select_least_loaded(workers)
            return workers[0]

    def _select_round_robin(self, workers: List[AgentWorker]) -> AgentWorker:
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker

    def _select_capability_match(
        self, workers: List[AgentWorker], task: Task
    ) -> Optional[AgentWorker]:
        candidates = [w for w in workers if w.can_handle(task)]
        if not candidates:
            return None
        # 在匹配者中选择负载最低的
        return min(candidates, key=lambda w: len(w.task_pool.tasks))

    def _select_least_loaded(self, workers: List[AgentWorker]) -> AgentWorker:
        return min(workers, key=lambda w: len(w.task_pool.tasks))

    def assign_task(
        self,
        task: Task,
        action: Optional[Callable] = None,
        strategy: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """将任务分配给合适的 worker 并立即执行。"""
        original_strategy = self._strategy
        if strategy is not None:
            self.set_strategy(strategy)
        try:
            worker = self.select_worker(task)
            if worker is None:
                return None
            worker.task_pool.update([task])
            result = worker.execute_task(task, action=action)
            return result
        finally:
            if strategy is not None:
                self.set_strategy(original_strategy)

    def run_mission(
        self,
        tasks: List[Task],
        actions: Optional[Dict[str, Callable]] = None,
        strategy: Optional[str] = None,
        mission_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """自驱动执行一组任务（mission），返回聚合结果。

        支持为不同任务名指定不同 action；未指定则使用 task.action。
        """
        mission_id = mission_id or f"mission_{uuid.uuid4().hex[:8]}"
        actions = actions or {}
        results: List[Dict[str, Any]] = []
        start = time.time()

        for task in tasks:
            action = actions.get(task.name)
            result = self.assign_task(task, action=action, strategy=strategy)
            if result is not None:
                results.append(result)

        completed = sum(1 for r in results if r.get('success'))
        summary = {
            'mission_id': mission_id,
            'total_tasks': len(tasks),
            'completed_tasks': completed,
            'failed_tasks': len(tasks) - completed,
            'duration': time.time() - start,
            'results': results,
        }
        self._missions[mission_id] = summary
        return summary

    def get_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """查询指定 mission 的执行结果。"""
        return self._missions.get(mission_id)

    def list_missions(self) -> List[str]:
        """返回所有 mission ID。"""
        return list(self._missions.keys())

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """聚合所有 worker 的统计信息。"""
        with self._lock:
            worker_stats = [w.get_stats() for w in self._workers.values()]
            total = sum(s['total_executions'] for s in worker_stats)
            success = sum(s['success_executions'] for s in worker_stats)
            return {
                'worker_count': len(worker_stats),
                'total_executions': total,
                'success_executions': success,
                'overall_success_rate': success / total if total > 0 else 0.0,
                'workers': worker_stats,
            }

    def clear_missions(self) -> None:
        """清空 mission 历史。"""
        self._missions.clear()
