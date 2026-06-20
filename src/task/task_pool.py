import abc
import threading
import time
from typing import Any, Dict, List, Optional

from src.core.constants import TaskType

from .task_definition import Task
from .task_generator import TaskGenerator
from .task_scorer import TaskScorer
from .task_selector import TaskSelector


class BaseTaskPool(abc.ABC):
    """Abstract base class for task pool management.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    @abc.abstractmethod
    def update(self, new_tasks: List[Task]) -> None:
        """Update the pool with new tasks.

        :param new_tasks: 新任务列表
        :type new_tasks: List[Task]
        :return: 无
        :rtype: None
        """
        pass

    @abc.abstractmethod
    def select_task(self) -> Optional[Task]:
        """Select a task from the pool.

        :param: 无
        :type: 无
        :return: 选中的任务或 None
        :rtype: Optional[Task]
        """
        pass


class TaskPool(BaseTaskPool):
    """Concrete task pool implementation.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    def __init__(self, max_size: int = 50, max_age_hours: float = 24.0) -> None:
        """Initialize the task pool.

        :param max_size: 任务池最大容量
        :type max_size: int
        :param max_age_hours: 任务最大存活时间（小时）
        :type max_age_hours: float
        :return: 无
        :rtype: None
        """
        self.tasks: Dict[str, Task] = {}
        self.max_size = self._sanitize_max_size(max_size)
        self.max_age_hours = max_age_hours if isinstance(max_age_hours, (int, float)) and max_age_hours > 0 else 24.0
        self.generator = TaskGenerator()
        self.scorer = TaskScorer()
        self.selector = TaskSelector()
        self._execution_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _sanitize_max_size(value: Any) -> int:
        """校正 max_size 参数，确保为正整数。

        :param value: 原始值。
        :type value: Any
        :return: 校正后的正整数。
        :rtype: int
        """
        try:
            v = int(value)
        except (TypeError, ValueError):
            return 50
        return v if v > 0 else 50

    def update(self, new_tasks: List[Task]) -> None:
        """Update the pool with new tasks.

        :param new_tasks: 新任务列表
        :type new_tasks: List[Task]
        :return: 无
        :rtype: None
        """
        with self._lock:
            if new_tasks is None:
                return
            self._add_new_tasks(new_tasks)
            self._enforce_size_limit()
            self._remove_stale_tasks()

    def _add_new_tasks(self, new_tasks: List[Task]) -> None:
        """Add new tasks to the pool.

        :param new_tasks: 新任务列表
        :type new_tasks: List[Task]
        :return: 无
        :rtype: None
        """
        for task in new_tasks:
            if task.id not in self.tasks:
                self.tasks[task.id] = task

    def score_all(self, state: Dict[str, Any], gap_vector: Dict[str, float]) -> Dict[str, float]:
        """Score all tasks in the pool.

        :param state: 当前状态
        :type state: Dict[str, Any]
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 任务 ID 到得分的映射
        :rtype: Dict[str, float]
        """
        with self._lock:
            return self.scorer.score_all(list(self.tasks.values()), state, gap_vector)

    def select_task(self) -> Optional[Task]:
        """Select a task from the pool.

        :param: 无
        :type: 无
        :return: 选中的任务或 None
        :rtype: Optional[Task]
        """
        with self._lock:
            return self.selector.select(list(self.tasks.values()))

    def mark_executed(self, task_id: str, success: bool = True, reward: float = 0.0) -> None:
        """Mark a task as executed.

        :param task_id: 任务 ID
        :type task_id: str
        :param success: 是否执行成功
        :type success: bool
        :param reward: 执行奖励
        :type reward: float
        :return: 无
        :rtype: None
        """
        with self._lock:
            if task_id not in self.tasks:
                return
            self.tasks[task_id].mark_executed(success, reward)
            self._record_execution(task_id, success, reward)

    def _record_execution(self, task_id: str, success: bool, reward: float) -> None:
        """Record an execution entry.

        :param task_id: 任务 ID
        :type task_id: str
        :param success: 是否执行成功
        :type success: bool
        :param reward: 执行奖励
        :type reward: float
        :return: 无
        :rtype: None
        """
        self._execution_history.append({
            'task_id': task_id,
            'success': success,
            'reward': reward,
            'timestamp': time.time(),
        })

    def freeze_task(self, task_id: str) -> None:
        """Freeze a task by ID.

        :param task_id: 任务 ID
        :type task_id: str
        :return: 无
        :rtype: None
        """
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].freeze()

    def unfreeze_task(self, task_id: str) -> None:
        """Unfreeze a task by ID.

        :param task_id: 任务 ID
        :type task_id: str
        :return: 无
        :rtype: None
        """
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].unfreeze()

    def remove_task(self, task_id: str) -> bool:
        """Remove a task by ID.

        :param task_id: 任务 ID
        :type task_id: str
        :return: 是否成功移除
        :rtype: bool
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            del self.tasks[task_id]
            return True

    def get_tasks_by_type(self, task_type: TaskType) -> List[Task]:
        """Return tasks filtered by type.

        :param task_type: 任务类型
        :type task_type: TaskType
        :return: 过滤后的任务列表
        :rtype: List[Task]
        """
        with self._lock:
            return [t for t in self.tasks.values() if t.type == task_type]

    def get_frozen_tasks(self) -> List[Task]:
        """Return all frozen tasks.

        :param: 无
        :type: 无
        :return: 已冻结的任务列表
        :rtype: List[Task]
        """
        with self._lock:
            return [t for t in self.tasks.values() if t.frozen]

    def get_top_tasks(self, n: int = 5) -> List[Task]:
        """Return top N ranked tasks.

        :param n: 返回的任务数量
        :type n: int
        :return: 排名前 N 的任务列表
        :rtype: List[Task]
        """
        with self._lock:
            return self.scorer.rank_tasks(list(self.tasks.values()))[:n]

    def _enforce_size_limit(self) -> None:
        """Remove lowest-ranked tasks if pool exceeds max size.

        新任务保护：当有老任务可驱逐时，优先驱逐老任务，避免新任务因score=0被立即删除。
        全为新任务时回退到原始行为（按排名驱逐）。

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        if len(self.tasks) <= self.max_size:
            return
        now = time.time()
        # 修复阶段120: 计算需要驱逐的数量，用[-excess:]而非[max_size:]避免子集为空
        excess = len(self.tasks) - self.max_size
        # 修复Bug3: 优先驱逐老任务（创建>5秒或已执行过），保护新任务
        old_tasks = [t for t in self.tasks.values()
                     if (now - t.created_at) >= 5.0 or t.execution_count > 0]
        if len(old_tasks) >= excess:
            # 有足够的老任务，从老任务中选最低分的excess个
            sorted_old = self.scorer.rank_tasks(old_tasks)
            to_remove = sorted_old[-excess:]
        else:
            # 老任务不够，从所有任务中按排名驱逐最低分的excess个
            sorted_all = self.scorer.rank_tasks(list(self.tasks.values()))
            to_remove = sorted_all[-excess:]
        for task in to_remove:
            del self.tasks[task.id]

    def _remove_stale_tasks(self) -> None:
        """Remove stale tasks that have never been executed.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        now = time.time()
        stale_ids = self._find_stale_ids(now)
        for task_id in stale_ids:
            del self.tasks[task_id]

    def _find_stale_ids(self, now: float) -> List[str]:
        """Find stale task IDs.

        :param now: 当前时间戳
        :type now: float
        :return: 过期任务 ID 列表
        :rtype: List[str]
        """
        stale_ids = []
        for task_id, task in self.tasks.items():
            age_hours = (now - task.created_at) / 3600
            if age_hours > self.max_age_hours and task.execution_count == 0:
                stale_ids.append(task_id)
        return stale_ids

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the task pool.

        :param: 无
        :type: 无
        :return: 任务池统计信息
        :rtype: Dict[str, Any]
        """
        with self._lock:
            total = len(self.tasks)
            frozen = self._count_frozen()
            return self._build_stats(total, frozen)

    def _build_stats(self, total: int, frozen: int) -> Dict[str, Any]:
        """Build the statistics dictionary.

        :param total: 任务总数
        :type total: int
        :param frozen: 冻结任务数
        :type frozen: int
        :return: 统计信息字典
        :rtype: Dict[str, Any]
        """
        by_type = self._count_by_type()
        return {
            'total_tasks': total,
            'frozen_tasks': frozen,
            'active_tasks': total - frozen,
            'by_type': by_type,
            'total_executions': len(self._execution_history),
        }

    def _count_frozen(self) -> int:
        """Count frozen tasks.

        :param: 无
        :type: 无
        :return: 冻结任务数量
        :rtype: int
        """
        return len([t for t in self.tasks.values() if t.frozen])

    def _count_by_type(self) -> Dict[str, int]:
        """Count tasks by type.

        :param: 无
        :type: 无
        :return: 任务类型到数量的映射
        :rtype: Dict[str, int]
        """
        by_type = {}
        for task in self.tasks.values():
            by_type[task.type.value] = by_type.get(task.type.value, 0) + 1
        return by_type
