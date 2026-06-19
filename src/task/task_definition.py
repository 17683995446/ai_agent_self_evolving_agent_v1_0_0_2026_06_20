import uuid
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from src.core.constants import TaskType


@dataclass
class Task:
    """Data class for a task.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        """Initialize id if it is None.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        if self.id is None:
            object.__setattr__(self, 'id', f"task_{uuid.uuid4().hex[:8]}")

    type: TaskType = TaskType.SAFE
    name: str = "unnamed"
    description: str = ""
    action: Optional[Callable] = None
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    score: float = 0.0
    execution_count: int = 0
    success_count: int = 0
    last_executed: Optional[float] = None
    frozen: bool = False
    avg_reward: float = 0.0

    @property
    def success_rate(self) -> float:
        """Return the success rate of the task.

        :param: 无
        :type: 无
        :return: 任务的成功率
        :rtype: float
        """
        if self.execution_count == 0:
            return 0.5
        return self.success_count / self.execution_count

    @property
    def is_exploration(self) -> bool:
        """Return whether the task is an exploration task.

        :param: 无
        :type: 无
        :return: 是否为探索任务
        :rtype: bool
        """
        return self.type == TaskType.EXPLORE

    def mark_executed(self, success: bool = True, reward: float = 0.0) -> None:
        """Mark the task as executed with success and reward.

        :param success: 是否执行成功
        :type success: bool
        :param reward: 执行奖励
        :type reward: float
        :return: 无
        :rtype: None
        """
        self.execution_count += 1
        if success:
            self.success_count += 1
        self.last_executed = time.time()
        if self.avg_reward == 0.0:
            self.avg_reward = reward
        else:
            self.avg_reward = 0.7 * self.avg_reward + 0.3 * reward

    def freeze(self) -> None:
        """Freeze the task to prevent execution.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the task to allow execution.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        self.frozen = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the task to a dictionary.

        :param: 无
        :type: 无
        :return: 任务的字典表示
        :rtype: Dict[str, Any]
        """
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'description': self.description,
            'params': self.params,
            'created_at': self.created_at,
            'score': self.score,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'last_executed': self.last_executed,
            'frozen': self.frozen,
            'avg_reward': self.avg_reward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Deserialize a task from a dictionary.

        :param data: 包含任务数据的字典
        :type data: Dict[str, Any]
        :return: 反序列化后的 Task 实例
        :rtype: Task
        """
        task = cls(
            id=data.get('id', f"task_{uuid.uuid4().hex[:8]}"),
            type=TaskType(data.get('type', 'safe')),
            name=data.get('name', 'unnamed'),
            description=data.get('description', ''),
            params=data.get('params', {}),
            created_at=data.get('created_at', time.time()),
            score=data.get('score', 0.0),
            execution_count=data.get('execution_count', 0),
            success_count=data.get('success_count', 0),
            last_executed=data.get('last_executed'),
            frozen=data.get('frozen', False),
            avg_reward=data.get('avg_reward', 0.0),
        )
        return task
