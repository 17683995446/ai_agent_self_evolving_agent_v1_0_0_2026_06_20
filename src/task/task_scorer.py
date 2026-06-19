import abc
import time
from typing import Dict, Any, List

from .task_definition import Task


class BaseTaskScorer(abc.ABC):
    """Abstract base class for task scoring.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    @abc.abstractmethod
    def score_task(self, task: Task, state: Dict[str, Any], gap_vector: Dict[str, float]) -> float:
        """Score a single task.

        :param task: 待评分的任务
        :type task: Task
        :param state: 当前状态
        :type state: Dict[str, Any]
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 任务得分
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def score_all(self, tasks: List[Task], state: Dict[str, Any], gap_vector: Dict[str, float]) -> Dict[str, float]:
        """Score all tasks and return a mapping of task ID to score.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :param state: 当前状态
        :type state: Dict[str, Any]
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 任务 ID 到得分的映射
        :rtype: Dict[str, float]
        """
        pass


class TaskScorer(BaseTaskScorer):
    """Concrete task scorer implementation.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    def __init__(self) -> None:
        """Initialize the task scorer.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        self.type_match_weight = 0.4
        self.success_bonus_weight = 0.3
        self.freshness_weight = 0.2
        self.time_decay_weight = 0.1

    def score_task(self, task: Task, state: Dict[str, Any], gap_vector: Dict[str, float]) -> float:
        """Score a single task.

        :param task: 待评分的任务
        :type task: Task
        :param state: 当前状态
        :type state: Dict[str, Any]
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 任务得分
        :rtype: float
        """
        if task.frozen:
            return -1.0
        task.score = self._compute_score(task, gap_vector)
        return task.score

    def _compute_score(self, task: Task, gap_vector: Dict[str, float]) -> float:
        """Compute the raw score for a task.

        :param task: 待评分的任务
        :type task: Task
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 原始得分
        :rtype: float
        """
        type_match = self._compute_type_match(task, gap_vector)
        success_bonus = self._compute_success_bonus(task)
        freshness = self._compute_freshness(task)
        time_decay = self._compute_time_decay(task)
        return self._assemble_score(type_match, success_bonus, freshness, time_decay)

    def _compute_type_match(self, task: Task, gap_vector: Dict[str, float]) -> float:
        """Compute type match score from gap vector.

        :param task: 待评分的任务
        :type task: Task
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 类型匹配得分
        :rtype: float
        """
        return gap_vector.get(task.type.value + '_gap', 0.0)

    def _compute_success_bonus(self, task: Task) -> float:
        """Compute success bonus for a task.

        :param task: 待评分的任务
        :type task: Task
        :return: 成功奖励得分
        :rtype: float
        """
        return task.success_rate * self.success_bonus_weight

    def _compute_freshness(self, task: Task) -> float:
        """Compute freshness score for a task.

        :param task: 待评分的任务
        :type task: Task
        :return: 新鲜度得分
        :rtype: float
        """
        return 1.0 if task.execution_count == 0 else 0.5

    def _compute_time_decay(self, task: Task) -> float:
        """Compute time decay factor for a task.

        :param task: 待评分的任务
        :type task: Task
        :return: 时间衰减因子
        :rtype: float
        """
        if task.last_executed is None:
            return 1.0
        hours_since = (time.time() - task.last_executed) / 3600
        return max(0.3, 1.0 - hours_since * 0.1)

    def _assemble_score(
        self,
        type_match: float,
        success_bonus: float,
        freshness: float,
        time_decay: float,
    ) -> float:
        """Assemble final score from components.

        :param type_match: 类型匹配得分
        :type type_match: float
        :param success_bonus: 成功奖励得分
        :type success_bonus: float
        :param freshness: 新鲜度得分
        :type freshness: float
        :param time_decay: 时间衰减因子
        :type time_decay: float
        :return: 最终得分
        :rtype: float
        """
        score = (type_match * self.type_match_weight +
                 success_bonus +
                 freshness * self.freshness_weight) * time_decay
        return round(score, 4)

    def score_all(self, tasks: List[Task], state: Dict[str, Any], gap_vector: Dict[str, float]) -> Dict[str, float]:
        """Score all tasks and return a mapping of task ID to score.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :param state: 当前状态
        :type state: Dict[str, Any]
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 任务 ID 到得分的映射
        :rtype: Dict[str, float]
        """
        scores = {}
        for task in tasks:
            task.score = self.score_task(task, state, gap_vector)
            scores[task.id] = task.score
        return scores

    def rank_tasks(self, tasks: List[Task]) -> List[Task]:
        """Rank tasks by score in descending order.

        :param tasks: 任务列表
        :type tasks: List[Task]
        :return: 按得分降序排列的任务列表
        :rtype: List[Task]
        """
        return sorted(tasks, key=lambda t: t.score, reverse=True)
