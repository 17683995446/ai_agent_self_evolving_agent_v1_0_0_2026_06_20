import threading
from typing import Dict, Any, List, Optional

from .skill import Skill


class SkillRegistry:
    """技能注册中心：管理 Skill 的注册、查询、匹配与执行。

    支持按能力标签检索，按历史成功率选择最佳技能，并提供聚合统计。
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._lock = threading.RLock()

    def register(self, skill: Skill) -> bool:
        """注册技能。同名技能已存在时返回 False，否则返回 True。"""
        with self._lock:
            if skill.name in self._skills:
                return False
            self._skills[skill.name] = skill
            return True

    def unregister(self, name: str) -> bool:
        """注销技能。"""
        with self._lock:
            return self._skills.pop(name, None) is not None

    def get(self, name: str) -> Optional[Skill]:
        """按名称获取技能。"""
        with self._lock:
            return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """返回已注册技能名称列表。"""
        with self._lock:
            return list(self._skills.keys())

    def find_by_capability(self, capability: str) -> List[Skill]:
        """查找具备指定能力标签的技能。"""
        with self._lock:
            return [s for s in self._skills.values() if capability in s.capabilities]

    def find_for_task(self, task) -> List[Skill]:
        """查找可以处理指定任务的全部技能。"""
        with self._lock:
            return [s for s in self._skills.values() if s.can_handle(task)]

    def select_best(self, task) -> Optional[Skill]:
        """为任务选择历史成功率最高的技能。

        若任务无能力要求且无特殊技能匹配，返回任意可用技能作为默认占位。
        """
        candidates = self.find_for_task(task)
        if not candidates:
            return None
        return max(candidates, key=lambda s: (s.success_rate, s.usage_count))

    def execute(self, task) -> Optional[Dict[str, Any]]:
        """自动选择最佳技能并执行任务。"""
        skill = self.select_best(task)
        if skill is None:
            return None
        return skill.execute(task)

    def update_skill(
        self,
        name: str,
        action=None,
        description: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> bool:
        """更新已注册技能的 action / 描述 / 能力标签。"""
        with self._lock:
            skill = self._skills.get(name)
            if skill is None:
                return False
            if action is not None:
                skill.action = action
            if description is not None:
                skill.description = description
            if capabilities is not None:
                skill.capabilities = list(capabilities)
            return True

    def to_dict(self) -> Dict[str, Any]:
        """返回注册中心聚合统计。"""
        with self._lock:
            skills = list(self._skills.values())
            total_usage = sum(s.usage_count for s in skills)
            total_success = sum(s.success_count for s in skills)
            total_failure = sum(s.failure_count for s in skills)
            return {
                'skill_count': len(skills),
                'total_usage': total_usage,
                'total_success': total_success,
                'total_failure': total_failure,
                'overall_success_rate': (
                    total_success / (total_success + total_failure)
                    if (total_success + total_failure) > 0
                    else 0.0
                ),
                'skills': [s.to_dict() for s in skills],
            }
