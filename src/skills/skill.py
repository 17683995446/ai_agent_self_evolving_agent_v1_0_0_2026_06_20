import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class Skill:
    """可执行技能单元，可被 AgentWorker 调用并在执行过程中自增强。

    :param name: 技能名称（唯一标识）
    :param description: 技能描述
    :param capabilities: 所需能力标签
    :param action: 执行函数，签名为 action(task) -> Any
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        action: Optional[Callable] = None,
    ) -> None:
        self.skill_id = f"skill_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.description = description
        self.capabilities = list(capabilities) if capabilities else []
        self.action = action
        self.success_count = 0
        self.failure_count = 0
        self.usage_count = 0
        self.created_at = time.time()
        self.last_used_at: Optional[float] = None

    def can_handle(self, task) -> bool:
        """判断该技能能否处理任务（任务需具备全部所需能力，或技能无特殊要求）。"""
        required = getattr(task, 'params', {}).get('required_capabilities', [])
        if not required and not self.capabilities:
            return True
        if not required:
            return True
        return all(req in self.capabilities for req in required)

    def execute(self, task) -> Dict[str, Any]:
        """执行技能并返回结果。统计由外部学习器统一更新。"""
        start = time.time()
        self.usage_count += 1
        self.last_used_at = time.time()
        if self.action is None:
            return {
                'success': True,
                'result': f"skill {self.name} placeholder",
                'duration': time.time() - start,
            }
        try:
            result = self.action(task)
            return {'success': True, 'result': result, 'duration': time.time() - start}
        except Exception as e:
            return {'success': False, 'result': str(e), 'duration': time.time() - start}

    def update_stats(self, success: bool) -> None:
        """根据执行结果更新成功率统计。"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    @property
    def success_rate(self) -> float:
        """计算历史成功率。"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            'skill_id': self.skill_id,
            'name': self.name,
            'description': self.description,
            'capabilities': self.capabilities,
            'success_rate': self.success_rate,
            'usage_count': self.usage_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'created_at': self.created_at,
            'last_used_at': self.last_used_at,
        }
