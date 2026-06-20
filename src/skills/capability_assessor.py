"""自我评估与能力差距分析。

评估 Agent 当前的技能覆盖率与能力矩阵，识别能力差距并生成新技能/任务建议，
形成"评估 → 识别差距 → 生成建议 → 学习填充"的自进化闭环。

第一性原理：
- 自进化 Agent 必须知道自己能做什么、不能做什么
- 能力差距 = 任务需求能力 - 已有技能/worker 覆盖能力
- 识别差距后应生成可操作的建议：创建新技能、练习弱技能、弃用坏技能
- 评估应覆盖：能力覆盖矩阵、成功率分布、未使用技能、低效技能
"""

from typing import Any, Dict, List, Optional, Set

from .skill_registry import SkillRegistry


class CapabilityAssessor:
    """评估当前技能覆盖率与能力矩阵。

    :param underperform_threshold: 成功率低于此值的技能视为低效
    :param unused_threshold: 使用次数低于此值视为未充分使用
    """

    def __init__(
        self,
        underperform_threshold: float = 0.3,
        unused_threshold: int = 0,
    ) -> None:
        self.underperform_threshold = max(0.0, min(1.0, underperform_threshold))
        self.unused_threshold = max(0, unused_threshold)

    def assess(
        self,
        registry: SkillRegistry,
        worker_capabilities: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """评估技能库与 worker 的能力覆盖。

        :param registry: 技能注册中心
        :param worker_capabilities: 各 worker 的能力标签列表
        :return: 评估结果字典
        """
        skills = [registry.get(name) for name in registry.list_skills()]
        skills = [s for s in skills if s is not None]

        skill_caps: Set[str] = set()
        for skill in skills:
            skill_caps.update(skill.capabilities)

        worker_caps: Set[str] = set()
        for caps in (worker_capabilities or []):
            worker_caps.update(caps)

        total_caps = skill_caps | worker_caps

        # 每个能力的技能覆盖数
        cap_coverage: Dict[str, int] = {}
        for skill in skills:
            for cap in skill.capabilities:
                cap_coverage[cap] = cap_coverage.get(cap, 0) + 1

        # 低效技能（有使用记录但成功率低）
        underperforming = [
            s.name for s in skills
            if s.usage_count > 0
            and s.success_rate < self.underperform_threshold
        ]

        # 未使用技能
        unused = [
            s.name for s in skills
            if s.usage_count <= self.unused_threshold
        ]

        # 高效技能（成功率 > 0.8 且使用 >= 3 次）
        proficient = [
            s.name for s in skills
            if s.usage_count >= 3 and s.success_rate > 0.8
        ]

        # 平均成功率
        rated_skills = [s for s in skills if s.usage_count > 0]
        avg_success = (
            sum(s.success_rate for s in rated_skills) / len(rated_skills)
            if rated_skills else 0.0
        )

        return {
            'skill_count': len(skills),
            'skill_capabilities': sorted(skill_caps),
            'worker_capabilities': sorted(worker_caps),
            'total_capabilities': sorted(total_caps),
            'capability_coverage': cap_coverage,
            'avg_success_rate': avg_success,
            'underperforming_skills': underperforming,
            'unused_skills': unused,
            'proficient_skills': proficient,
        }


class GapAnalyzer:
    """识别能力差距并生成新技能/任务建议。

    :param min_coverage: 能力最少需要的技能覆盖数，低于此值视为不足
    """

    def __init__(self, min_coverage: int = 1) -> None:
        self.min_coverage = max(1, min_coverage)

    def analyze(
        self,
        assessment: Dict[str, Any],
        required_capabilities: List[str],
    ) -> Dict[str, Any]:
        """分析能力差距并生成建议。

        :param assessment: CapabilityAssessor 的评估结果
        :param required_capabilities: 任务需求的能力列表
        :return: 差距分析结果，包含 missing/understaffed/suggestions
        """
        available = set(assessment.get('total_capabilities', []))
        required = set(required_capabilities)

        missing = sorted(required - available)

        coverage = assessment.get('capability_coverage', {})
        understaffed = sorted(
            cap for cap in (required & available)
            if coverage.get(cap, 0) < self.min_coverage
        )

        suggestions: List[Dict[str, Any]] = []

        # 为缺失能力生成创建技能建议
        for cap in missing:
            suggestions.append({
                'type': 'create_skill',
                'capability': cap,
                'reason': f"no skill or worker covers '{cap}'",
                'priority': 'high',
            })

        # 为覆盖不足能力生成补充建议
        for cap in understaffed:
            suggestions.append({
                'type': 'augment_skill',
                'capability': cap,
                'reason': f"only {coverage.get(cap, 0)} skill(s) cover '{cap}'",
                'priority': 'medium',
            })

        # 为低效技能生成练习建议
        for name in assessment.get('underperforming_skills', []):
            suggestions.append({
                'type': 'practice_skill',
                'skill_name': name,
                'reason': f"skill '{name}' success rate below threshold",
                'priority': 'medium',
            })

        # 为未使用技能生成探索建议
        for name in assessment.get('unused_skills', []):
            suggestions.append({
                'type': 'explore_skill',
                'skill_name': name,
                'reason': f"skill '{name}' has never been used",
                'priority': 'low',
            })

        return {
            'required_capabilities': sorted(required),
            'available_capabilities': sorted(available),
            'missing_capabilities': missing,
            'understaffed_capabilities': understaffed,
            'gap_count': len(missing) + len(understaffed),
            'suggestions': suggestions,
        }

    def suggest_tasks(
        self,
        gap_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """将差距分析转化为可执行的任务建议。

        :return: 任务建议列表，每条包含 task_name, capability, rationale
        """
        tasks: List[Dict[str, Any]] = []
        for suggestion in gap_analysis.get('suggestions', []):
            if suggestion['type'] == 'create_skill':
                tasks.append({
                    'task_name': f"learn_{suggestion['capability']}",
                    'capability': suggestion['capability'],
                    'rationale': suggestion['reason'],
                    'priority': suggestion['priority'],
                })
            elif suggestion['type'] == 'augment_skill':
                tasks.append({
                    'task_name': f"augment_{suggestion['capability']}",
                    'capability': suggestion['capability'],
                    'rationale': suggestion['reason'],
                    'priority': suggestion['priority'],
                })
            elif suggestion['type'] == 'practice_skill':
                tasks.append({
                    'task_name': f"practice_{suggestion['skill_name']}",
                    'capability': None,
                    'rationale': suggestion['reason'],
                    'priority': suggestion['priority'],
                })
            elif suggestion['type'] == 'explore_skill':
                tasks.append({
                    'task_name': f"explore_{suggestion['skill_name']}",
                    'capability': None,
                    'rationale': suggestion['reason'],
                    'priority': suggestion['priority'],
                })
        return tasks
