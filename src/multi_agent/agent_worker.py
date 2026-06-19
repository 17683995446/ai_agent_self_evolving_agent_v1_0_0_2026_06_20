import time
import uuid
from typing import Dict, Any, List, Optional, Callable

from src.task.task_pool import TaskPool
from src.task.task_definition import Task
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_learner import SkillLearner


class AgentWorker:
    """轻量级 Agent 工作节点，可独立执行任务并维护本地任务池。

    :param worker_id: 工作节点唯一标识
    :param capabilities: 能力标签集合，用于编排器做能力匹配
    :param task_pool: 可选的任务池实例
    :param skill_registry: 可选的技能注册中心，启用后支持 use_skill 自动匹配
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        task_pool: Optional[TaskPool] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ) -> None:
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.capabilities: List[str] = list(capabilities) if capabilities else []
        self.task_pool = task_pool or TaskPool()
        self.skill_registry = skill_registry
        # 若提供了 registry 则复用，否则 learner 为 None，use_skill 时按需创建
        self.skill_learner: Optional[SkillLearner] = (
            SkillLearner(self.skill_registry) if self.skill_registry is not None else None
        )
        self._execution_log: List[Dict[str, Any]] = []
        self._busy = False

    def attach_skill_registry(self, registry: SkillRegistry) -> None:
        """绑定技能注册中心与学习器，启用 use_skill 自动匹配。"""
        self.skill_registry = registry
        self.skill_learner = SkillLearner(registry)

    def add_capability(self, capability: str) -> None:
        """为 worker 增加能力标签。"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def has_capability(self, capability: str) -> bool:
        """判断是否具备指定能力。"""
        return capability in self.capabilities

    def can_handle(self, task: Task) -> bool:
        """根据任务参数中的 required_capabilities 判断能否处理。"""
        required = task.params.get('required_capabilities', [])
        if not required:
            return True
        return all(req in self.capabilities for req in required)

    def use_skill(self, task: Task) -> Optional[Dict[str, Any]]:
        """自动从技能注册表中选择最佳技能并执行任务。

        - 若无注册表或无匹配技能，返回 None。
        - 执行后通过 SkillLearner 吸收反馈，更新技能统计。
        """
        if self.skill_registry is None or self.skill_learner is None:
            return None
        skill = self.skill_registry.select_best(task)
        if skill is None:
            return None
        result = skill.execute(task)
        # 学习器统一更新统计，避免重复计数
        self.skill_learner.feedback(skill.name, result.get('success', False))
        # 复用 execute_task 的记录与任务状态更新逻辑
        return self._record_execution(task, result)

    def execute_task(self, task: Task, action: Optional[Callable] = None) -> Dict[str, Any]:
        """执行单个任务并返回结果。

        若 task.action 或传入 action 不为空则调用，否则视为成功占位。
        """
        self._busy = True
        start = time.time()
        try:
            handler = action or task.action
            if handler is not None:
                raw = handler(task)
                if isinstance(raw, dict) and 'success' in raw:
                    success = bool(raw['success'])
                    result = raw.get('result', raw)
                else:
                    success = True
                    result = raw
            else:
                result = f"executed by {self.worker_id}"
                success = True
            reward = 1.0 if success else -1.0
        except Exception as e:
            result = str(e)
            success = False
            reward = -1.0
        finally:
            self._busy = False

        duration = time.time() - start
        task.mark_executed(success=success, reward=reward)
        record = {
            'worker_id': self.worker_id,
            'task_id': task.id,
            'task_name': task.name,
            'success': success,
            'reward': reward,
            'duration': duration,
            'result': result,
            'timestamp': time.time(),
        }
        self._execution_log.append(record)
        return record

    def _record_execution(self, task: Task, skill_result: Dict[str, Any]) -> Dict[str, Any]:
        """将技能执行结果转换为标准执行记录并更新任务状态。"""
        self._busy = True
        success = bool(skill_result.get('success', False))
        result = skill_result.get('result', skill_result)
        duration = skill_result.get('duration', 0.0)
        reward = 1.0 if success else -1.0
        self._busy = False
        task.mark_executed(success=success, reward=reward)
        record = {
            'worker_id': self.worker_id,
            'task_id': task.id,
            'task_name': task.name,
            'success': success,
            'reward': reward,
            'duration': duration,
            'result': result,
            'timestamp': time.time(),
            'skill_used': True,
        }
        self._execution_log.append(record)
        return record

    def is_busy(self) -> bool:
        """当前是否正在执行任务。"""
        return self._busy

    def get_stats(self) -> Dict[str, Any]:
        """返回 worker 统计信息。"""
        total = len(self._execution_log)
        success_count = sum(1 for r in self._execution_log if r['success'])
        skill_used_count = sum(1 for r in self._execution_log if r.get('skill_used'))
        return {
            'worker_id': self.worker_id,
            'capabilities': self.capabilities,
            'total_executions': total,
            'success_executions': success_count,
            'success_rate': success_count / total if total > 0 else 0.0,
            'skill_used_count': skill_used_count,
            'skill_registry_attached': self.skill_registry is not None,
            'busy': self._busy,
            'task_pool_size': len(self.task_pool.tasks),
        }

