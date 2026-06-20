"""技能自动发现与组合。

从执行轨迹中发现重复的任务序列模式，并将高频模式组合为新的复合技能，
使 Agent 能够自主抽象出更高层次的技能，减少重复编排开销。

第一性原理：
- 反复出现的任务序列意味着存在可复用的高层操作
- 将序列固化为单一技能可降低编排成本并提升稳定性
- 复合技能的成功率由其组件的成功率联合决定
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .skill import Skill
from .skill_registry import SkillRegistry


class SkillDiscovery:
    """从执行轨迹中发现重复的任务序列模式。

    :param min_support: 模式最少出现次数，低于此值不视为频繁
    :param min_length: 模式最短长度（任务数）
    :param max_length: 模式最长长度，避免组合爆炸
    """

    def __init__(
        self,
        min_support: int = 2,
        min_length: int = 2,
        max_length: int = 5,
    ) -> None:
        self.min_support = max(1, min_support)
        self.min_length = max(2, min_length)
        self.max_length = max(self.min_length, max_length)

    def discover(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从执行记录中发现频繁任务序列模式。

        :param records: 执行记录列表，每条至少包含 ``task_name`` 字段
        :return: 模式列表，按 support 降序排列。每条包含：
                 - ``pattern``: 任务名序列
                 - ``support``: 出现次数
                 - ``avg_success``: 该模式出现的平均成功率
        """
        sequences = self._extract_sequences(records)
        if not sequences:
            return []

        pattern_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        for seq in sequences:
            length = len(seq)
            if length < self.min_length:
                continue
            upper = min(self.max_length, length)
            for n in range(self.min_length, upper + 1):
                for start in range(0, length - n + 1):
                    window = seq[start:start + n]
                    pattern_key = tuple(name for name, _ in window)
                    stat = pattern_stats.setdefault(
                        pattern_key, {'support': 0, 'successes': 0}
                    )
                    stat['support'] += 1
                    # 窗口内全部成功才算该次出现成功
                    if all(success for _, success in window):
                        stat['successes'] += 1

        results: List[Dict[str, Any]] = []
        for pattern, stat in pattern_stats.items():
            if stat['support'] < self.min_support:
                continue
            support = stat['support']
            avg_success = stat['successes'] / support if support > 0 else 0.0
            results.append({
                'pattern': list(pattern),
                'support': support,
                'avg_success': avg_success,
            })

        results.sort(key=lambda r: (r['support'], r['avg_success']), reverse=True)
        return results

    @staticmethod
    def _extract_sequences(records: List[Dict[str, Any]]) -> List[List[Tuple[str, bool]]]:
        """从记录中提取 (任务名, 是否成功) 序列，按 mission_id 分组（若存在）。"""
        if not records:
            return []

        grouped: Dict[str, List[Tuple[str, bool]]] = {}
        order: List[str] = []
        for rec in records:
            name = rec.get('task_name') or rec.get('skill_name')
            if not name:
                continue
            success = bool(rec.get('success', False))
            mission_id = rec.get('mission_id', rec.get('plan_id', '__default__'))
            if mission_id not in grouped:
                grouped[mission_id] = []
                order.append(mission_id)
            grouped[mission_id].append((name, success))

        return [grouped[mid] for mid in order if mid in grouped]


class CompositeFailure(Exception):
    """复合技能执行失败异常，携带各步骤执行详情。"""

    def __init__(self, message: str, details: List[Dict[str, Any]]) -> None:
        super().__init__(message)
        self.details = details


class SkillComposer:
    """将发现的频繁模式组合为新的复合技能。

    复合技能按顺序执行组件技能，任一组件失败则整体失败（抛出 CompositeFailure）。
    复合技能的能力标签为其全部组件能力标签的并集。

    :param registry: 技能注册中心，用于查找组件技能与注册复合技能
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def compose(
        self,
        pattern: List[str],
        name: Optional[str] = None,
        description: str = "",
    ) -> Optional[Skill]:
        """将一组技能名组合为新的复合技能并注册。

        :param pattern: 组件技能名序列（按执行顺序）
        :param name: 新技能名，默认自动生成 ``composite:a+b``
        :param description: 技能描述
        :return: 新注册的复合 Skill，若任一组件不存在则返回 None
        """
        if not pattern:
            return None

        components: List[Skill] = []
        for skill_name in pattern:
            skill = self.registry.get(skill_name)
            if skill is None:
                return None
            components.append(skill)

        composite_name = name or '+'.join(pattern)
        composite_caps = self._union_capabilities(components)
        action = self._build_composite_action(components)
        composite = Skill(
            name=composite_name,
            description=description or f"复合技能: {' -> '.join(pattern)}",
            capabilities=composite_caps,
            action=action,
        )
        # 若同名复合技能已存在则更新
        if not self.registry.register(composite):
            self.registry.update_skill(
                composite_name,
                action=action,
                description=description or f"复合技能: {' -> '.join(pattern)}",
                capabilities=composite_caps,
            )
        return composite

    def compose_from_records(
        self,
        records: List[Dict[str, Any]],
        discovery: Optional[SkillDiscovery] = None,
        top_k: int = 3,
    ) -> List[Skill]:
        """从执行记录中自动发现模式并组合为复合技能。

        :param records: 执行记录列表
        :param discovery: 可选的发现器实例，默认新建
        :param top_k: 最多组合前 K 个模式
        :return: 新组合的复合技能列表
        """
        discovery = discovery or SkillDiscovery()
        patterns = discovery.discover(records)
        composed: List[Skill] = []
        for entry in patterns[:top_k]:
            skill = self.compose(entry['pattern'])
            if skill is not None:
                composed.append(skill)
        return composed

    @staticmethod
    def _union_capabilities(components: List[Skill]) -> List[str]:
        """取全部组件能力标签的并集，保持首次出现顺序。"""
        seen: Dict[str, None] = {}
        for skill in components:
            for cap in skill.capabilities:
                seen[cap] = None
        return list(seen.keys())

    @staticmethod
    def _build_composite_action(components: List[Skill]) -> Callable:
        """构建复合 action：顺序执行组件，任一失败则抛出 CompositeFailure。"""

        def action(task):
            results: List[Dict[str, Any]] = []
            for skill in components:
                outcome = skill.execute(task)
                results.append({
                    'skill': skill.name,
                    'success': outcome.get('success', False),
                    'result': outcome.get('result'),
                })
                if not outcome.get('success', False):
                    raise CompositeFailure(
                        f"composite failed at {skill.name}",
                        results,
                    )
            return {
                'success': True,
                'result': f"composite succeeded with {len(components)} steps",
                'details': results,
            }

        return action
