from typing import Any, Callable, Dict, List, Optional

from .skill import Skill
from .skill_registry import SkillRegistry


class SkillLearner:
    """技能自学习器：根据执行反馈自动注册、更新与优化技能。

    - 支持从任务和 action 自动抽取新技能并注册。
    - 根据执行结果更新技能统计。
    - 基于成功率给出弃用 / 偏好 / 练习建议。
    """

    def __init__(self, registry: Optional[SkillRegistry] = None) -> None:
        self.registry = registry or SkillRegistry()

    def auto_register(
        self,
        task,
        action: Callable,
        description: str = "",
        capabilities: Optional[List[str]] = None,
    ) -> Skill:
        """从任务与 action 自动生成并注册技能。"""
        name = getattr(task, 'name', 'unnamed')
        required = (
            getattr(task, 'params', {}).get('required_capabilities', [])
            or capabilities
            or []
        )
        caps = list(required) if not capabilities else capabilities
        skill = Skill(name=name, description=description, capabilities=caps, action=action)
        if not self.registry.register(skill):
            # 若已存在则更新 action，便于技能自增强替换旧实现
            self.registry.update_skill(name, action=action)
        return skill

    def feedback(self, name: str, success: bool) -> bool:
        """根据一次执行反馈更新技能统计。"""
        skill = self.registry.get(name)
        if skill is None:
            return False
        skill.update_stats(success=success)
        return True

    def improve(self) -> Dict[str, Any]:
        """基于历史成功率给出技能优化建议。"""
        recommendations: Dict[str, List[str]] = {
            'deprecate': [],
            'preferred': [],
            'practice': [],
        }
        for skill in self.registry._skills.values():
            if skill.usage_count >= 5 and skill.success_rate < 0.3:
                recommendations['deprecate'].append(skill.name)
            elif skill.usage_count >= 3 and skill.success_rate > 0.8:
                recommendations['preferred'].append(skill.name)
            elif skill.usage_count < 3:
                recommendations['practice'].append(skill.name)
        return recommendations

    def suggest(self, task) -> Optional[Skill]:
        """为任务推荐最佳技能。"""
        return self.registry.select_best(task)

    def learn_from_record(self, record: Dict[str, Any]) -> bool:
        """从单条执行记录中提取反馈并更新对应技能统计。"""
        name = record.get('skill_name') or record.get('task_name')
        if not name:
            return False
        return self.feedback(name, record.get('success', False))
