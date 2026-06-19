from typing import Any, Callable, Dict, List, Optional

from src.task.task_definition import Task
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_learner import SkillLearner
from src.planning.long_task_planner import LongTaskPlanner
from src.memory.long_term import LongTermMemory


class MultiAgentLongTaskRunner:
    """多 Agent 长任务运行器：整合编排器、技能注册表、学习器与长任务规划器。

    - 支持将长任务分解为带能力要求的子任务。
    - 自动为子任务匹配已注册技能并通过 Orchestrator 分发给合适的 Agent。
    - 执行结束后自动用 SkillLearner 吸收反馈，持续自增强。
    """

    def __init__(
        self,
        orchestrator: MultiAgentOrchestrator,
        registry: Optional[SkillRegistry] = None,
        learner: Optional[SkillLearner] = None,
        memory: Optional[LongTermMemory] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.registry = registry or SkillRegistry()
        self.learner = learner or SkillLearner(self.registry)
        self.memory = memory

    def register_skill(
        self,
        name: str,
        action: Callable,
        capabilities: List[str],
        description: str = "",
    ) -> Any:
        """注册一个可被 Agent 调用的技能。"""
        placeholder = Task(name=name, params={'required_capabilities': capabilities})
        return self.learner.auto_register(
            placeholder,
            action=action,
            capabilities=capabilities,
            description=description,
        )

    def run(
        self,
        goal: Task,
        steps: List[Task],
        max_steps: int = 100,
        continue_on_failure: bool = False,
    ) -> Dict[str, Any]:
        """自驱动执行多 Agent 长任务。"""
        planner = LongTaskPlanner(
            goal,
            orchestrator=self.orchestrator,
            memory=self.memory,
            max_steps=max_steps,
            continue_on_failure=continue_on_failure,
        )
        for step in steps:
            skill = self.registry.select_best(step)

            def make_action(s=skill):
                def _action(t):
                    return s.execute(t)
                return _action

            action: Optional[Callable] = make_action() if skill is not None else None
            planner.add_step(step, action=action)

        summary = planner.run()
        self._learn(summary)
        return summary

    def _learn(self, summary: Dict[str, Any]) -> None:
        """从执行结果中提取技能反馈并更新学习器。"""
        for record in summary.get('results', []):
            self.learner.learn_from_record(record)

    def skill_stats(self) -> Dict[str, Any]:
        """返回技能注册表统计。"""
        return self.registry.to_dict()
