import abc
import random
from typing import List, Optional

from .task_definition import Task


class BaseTaskSelector(abc.ABC):
    """Abstract base class for task selection.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    @abc.abstractmethod
    def select(self, tasks: List[Task]) -> Optional[Task]:
        """Select a task from the given list.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :return: 选中的任务或 None
        :rtype: Optional[Task]
        """
        pass

    @abc.abstractmethod
    def adjust_ratio(self, recent_mae: float, is_emergency: bool = False,
                     is_novel_environment: bool = False) -> float:
        """Adjust the exploration ratio.

        :param recent_mae: 最近的平均绝对误差
        :type recent_mae: float
        :param is_emergency: 是否为紧急情况
        :type is_emergency: bool
        :param is_novel_environment: 是否为新环境
        :type is_novel_environment: bool
        :return: 调整后的探索比例
        :rtype: float
        """
        pass


class TaskSelector(BaseTaskSelector):
    """Concrete task selector with explore-exploit balance.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    def __init__(self, explore_ratio: float = 0.3) -> None:
        """Initialize the task selector.

        :param explore_ratio: 基础探索比例
        :type explore_ratio: float
        :return: 无
        :rtype: None
        """
        self.base_explore_ratio = explore_ratio
        self.current_ratio = explore_ratio
        self.stable_threshold = 0.05

    def select(self, tasks: List[Task]) -> Optional[Task]:
        """Select a task from the given list.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :return: 选中的任务或 None
        :rtype: Optional[Task]
        """
        valid_tasks = self._filter_valid(tasks)
        if not valid_tasks:
            return None
        return self._choose(valid_tasks)

    def _filter_valid(self, tasks: List[Task]) -> List[Task]:
        """Filter out frozen tasks.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :return: 过滤后的有效任务列表
        :rtype: List[Task]
        """
        if not tasks:
            return []
        return [t for t in tasks if not t.frozen]

    def _choose(self, tasks: List[Task]) -> Optional[Task]:
        """Choose between explore and exploit.

        :param tasks: 有效任务列表
        :type tasks: List[Task]
        :return: 选中的任务或 None
        :rtype: Optional[Task]
        """
        if random.random() < self.current_ratio:
            return self._explore(tasks)
        return self._exploit(tasks)

    def _exploit(self, tasks: List[Task]) -> Optional[Task]:
        """Select the highest-scoring task.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :return: 得分最高的任务或 None
        :rtype: Optional[Task]
        """
        if not tasks:
            return None
        return max(tasks, key=lambda t: t.score)

    def _explore(self, tasks: List[Task]) -> Optional[Task]:
        """Select a random low-scoring or any task.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :return: 随机选择的低分任务或任意任务
        :rtype: Optional[Task]
        """
        if not tasks:
            return None
        low_score_tasks = [t for t in tasks if t.score < 0.5]
        if low_score_tasks:
            return random.choice(low_score_tasks)
        return random.choice(tasks)

    def adjust_ratio(self, recent_mae: float, is_emergency: bool = False,
                     is_novel_environment: bool = False) -> float:
        """Adjust the exploration ratio based on environment conditions.

        :param recent_mae: 最近的平均绝对误差
        :type recent_mae: float
        :param is_emergency: 是否为紧急情况
        :type is_emergency: bool
        :param is_novel_environment: 是否为新环境
        :type is_novel_environment: bool
        :return: 调整后的探索比例
        :rtype: float
        """
        self.current_ratio = self._resolve_ratio(recent_mae, is_emergency, is_novel_environment)
        return self.current_ratio

    def _resolve_ratio(self, recent_mae: float, is_emergency: bool, is_novel_environment: bool) -> float:
        """Resolve the appropriate exploration ratio.

        :param recent_mae: 最近的平均绝对误差
        :type recent_mae: float
        :param is_emergency: 是否为紧急情况
        :type is_emergency: bool
        :param is_novel_environment: 是否为新环境
        :type is_novel_environment: bool
        :return: 合适的探索比例
        :rtype: float
        """
        if is_emergency:
            return 0.1
        if is_novel_environment:
            return 0.4
        if recent_mae < self.stable_threshold:
            return 0.5
        return self.base_explore_ratio

    def get_ratio(self) -> float:
        """Return the current exploration ratio.

        :param: 无
        :type: 无
        :return: 当前探索比例
        :rtype: float
        """
        return self.current_ratio
