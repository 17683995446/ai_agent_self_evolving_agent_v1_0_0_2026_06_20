import abc
from typing import Dict, List, Callable

from src.core.constants import TaskType, SAFE_GAP_THRESHOLD, GAIN_GAP_THRESHOLD, EFF_GAP_THRESHOLD
from .task_definition import Task


class BaseTaskGenerator(abc.ABC):
    """Abstract base class for task generation.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    @abc.abstractmethod
    def generate(self, gap_vector: Dict[str, float]) -> List[Task]:
        """Generate tasks based on the given gap vector.

        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 生成的任务列表
        :rtype: List[Task]
        """
        pass


class TaskGenerator(BaseTaskGenerator):
    """Concrete task generator based on gap vectors.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    def __init__(self) -> None:
        """Initialize the task generator.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        self._safe_actions = self._build_safe_actions()
        self._gain_actions = self._build_gain_actions()
        self._eff_actions = self._build_eff_actions()

    def _build_safe_actions(self) -> Dict[str, Callable]:
        """Build safe action templates.

        :param: 无
        :type: 无
        :return: 安全动作模板字典
        :rtype: Dict[str, Callable]
        """
        return {
            'cpu_monitor': self._dummy_action,
            'mem_monitor': self._dummy_action,
            'disk_monitor': self._dummy_action,
            'error_scan': self._dummy_action,
            'process_audit': self._dummy_action,
            'network_check': self._dummy_action,
        }

    def _build_gain_actions(self) -> Dict[str, Callable]:
        """Build gain action templates.

        :param: 无
        :type: 无
        :return: 增益动作模板字典
        :rtype: Dict[str, Callable]
        """
        return {
            'new_feature_explore': self._dummy_action,
            'pattern_mining': self._dummy_action,
            'correlation_test': self._dummy_action,
            'dimension_expand': self._dummy_action,
        }

    def _build_eff_actions(self) -> Dict[str, Callable]:
        """Build efficiency action templates.

        :param: 无
        :type: 无
        :return: 效率动作模板字典
        :rtype: Dict[str, Callable]
        """
        return {
            'model_prune': self._dummy_action,
            'cache_cleanup': self._dummy_action,
            'history_compress': self._dummy_action,
            'frequency_reduce': self._dummy_action,
        }

    def generate(self, gap_vector: Dict[str, float]) -> List[Task]:
        """Generate tasks based on the given gap vector.

        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :return: 生成的任务列表
        :rtype: List[Task]
        """
        tasks = []
        self._extend_gap_tasks(tasks, gap_vector, 'safe_gap', SAFE_GAP_THRESHOLD, self._generate_safe_tasks)
        self._extend_gap_tasks(tasks, gap_vector, 'gain_gap', GAIN_GAP_THRESHOLD, self._generate_gain_tasks)
        self._extend_gap_tasks(tasks, gap_vector, 'eff_gap', EFF_GAP_THRESHOLD, self._generate_eff_tasks)
        return tasks

    def _extend_gap_tasks(
        self,
        tasks: List[Task],
        gap_vector: Dict[str, float],
        gap_key: str,
        threshold: float,
        generator: Callable[[int], List[Task]],
    ) -> None:
        """Extend tasks list with tasks for a specific gap.

        :param tasks: 现有任务列表
        :type tasks: List[Task]
        :param gap_vector: 差距向量
        :type gap_vector: Dict[str, float]
        :param gap_key: 差距键名
        :type gap_key: str
        :param threshold: 阈值
        :type threshold: float
        :param generator: 任务生成器函数
        :type generator: Callable[[int], List[Task]]
        :return: 无
        :rtype: None
        """
        gap = gap_vector.get(gap_key, 0.0)
        tasks.extend(self._generate_for_gap(gap, threshold, generator))

    def _dummy_action(self) -> bool:
        """Return a dummy action result.

        :param: 无
        :type: 无
        :return: 虚拟动作结果
        :rtype: bool
        """
        return True

    def _generate_for_gap(
        self,
        gap: float,
        threshold: float,
        generator: Callable[[int], List[Task]],
    ) -> List[Task]:
        """Generate tasks if gap exceeds threshold.

        :param gap: 差距值
        :type gap: float
        :param threshold: 阈值
        :type threshold: float
        :param generator: 任务生成器函数
        :type generator: Callable[[int], List[Task]]
        :return: 生成的任务列表
        :rtype: List[Task]
        """
        if gap <= threshold:
            return []
        count = min(3, max(1, int(gap * 5)))
        return generator(count)

    def _generate_safe_tasks(self, count: int) -> List[Task]:
        """Generate safe tasks.

        :param count: 生成任务数量
        :type count: int
        :return: 安全任务列表
        :rtype: List[Task]
        """
        return self._generate_tasks(count, self._safe_actions, TaskType.SAFE, 'safe')

    def _generate_gain_tasks(self, count: int) -> List[Task]:
        """Generate gain tasks.

        :param count: 生成任务数量
        :type count: int
        :return: 增益任务列表
        :rtype: List[Task]
        """
        return self._generate_tasks(count, self._gain_actions, TaskType.GAIN, 'gain')

    def _generate_eff_tasks(self, count: int) -> List[Task]:
        """Generate efficiency tasks.

        :param count: 生成任务数量
        :type count: int
        :return: 效率任务列表
        :rtype: List[Task]
        """
        return self._generate_tasks(count, self._eff_actions, TaskType.EFF, 'eff')

    def _generate_tasks(
        self,
        count: int,
        actions: Dict[str, Callable],
        task_type: TaskType,
        label: str,
    ) -> List[Task]:
        """Build Task objects from action templates.

        :param count: 生成任务数量
        :type count: int
        :param actions: 动作模板字典
        :type actions: Dict[str, Callable]
        :param task_type: 任务类型
        :type task_type: TaskType
        :param label: 任务标签
        :type label: str
        :return: 构建的任务列表
        :rtype: List[Task]
        """
        tasks = []
        templates = list(actions.items())
        for i in range(min(count, len(templates))):
            name, action = templates[i]
            tasks.append(self._build_task(name, action, task_type, label))
        return tasks

    def _build_task(self, name: str, action: Callable, task_type: TaskType, label: str) -> Task:
        """Build a single Task object.

        :param name: 任务名称
        :type name: str
        :param action: 任务动作
        :type action: Callable
        :param task_type: 任务类型
        :type task_type: TaskType
        :param label: 任务标签
        :type label: str
        :return: 构建的 Task 实例
        :rtype: Task
        """
        return Task(
            type=task_type,
            name=name,
            description=f"Auto-generated {label} task: {name}",
            action=action,
            params={'priority': 'high'},
        )

    def add_safe_template(self, name: str, action: Callable) -> None:
        """Add a safe task template.

        :param name: 模板名称
        :type name: str
        :param action: 模板动作
        :type action: Callable
        :return: 无
        :rtype: None
        """
        self._safe_actions[name] = action

    def add_gain_template(self, name: str, action: Callable) -> None:
        """Add a gain task template.

        :param name: 模板名称
        :type name: str
        :param action: 模板动作
        :type action: Callable
        :return: 无
        :rtype: None
        """
        self._gain_actions[name] = action

    def add_eff_template(self, name: str, action: Callable) -> None:
        """Add an efficiency task template.

        :param name: 模板名称
        :type name: str
        :param action: 模板动作
        :type action: Callable
        :return: 无
        :rtype: None
        """
        self._eff_actions[name] = action
