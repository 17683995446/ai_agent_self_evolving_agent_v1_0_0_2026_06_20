import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable

from src.task.task_definition import Task
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)


class LongTaskPlanner:
    """长任务自驱动规划器：将目标拆解为子任务序列并自动推进执行。

    - 支持基于 Orchestrator 的多 Agent 分发，或本地 Worker 单点执行。
    - 支持 checkpoint / resume，断点续跑。
    - 支持失败重试与简单的 replan 回调扩展。
    """

    def __init__(
        self,
        goal: Task,
        orchestrator: Optional[MultiAgentOrchestrator] = None,
        worker: Optional[AgentWorker] = None,
        memory: Optional[LongTermMemory] = None,
        max_steps: int = 100,
        continue_on_failure: bool = False,
        replan_callback: Optional[Callable[['LongTaskPlanner', int, Dict[str, Any]], Optional[List[Task]]]] = None,
    ) -> None:
        self.plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        self.goal = goal
        self.orchestrator = orchestrator
        self.worker = worker or AgentWorker()
        self.memory = memory
        self.max_steps = max_steps
        self.continue_on_failure = continue_on_failure
        self.replan_callback = replan_callback

        self._steps: List[Task] = []
        self._actions: Dict[str, Callable] = {}
        self._retry_counts: Dict[str, int] = {}
        self._current_index = 0
        self._results: List[Dict[str, Any]] = []
        self._status = 'pending'

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for r in self._results if r.get('success'))

    @property
    def failed_steps(self) -> int:
        return sum(1 for r in self._results if not r.get('success'))

    def add_step(self, task: Task, action: Optional[Callable] = None, retries: int = 1) -> 'LongTaskPlanner':
        """添加子任务，可绑定执行函数与重试次数。"""
        self._steps.append(task)
        self._actions[task.id] = action
        self._retry_counts[task.id] = retries
        return self

    def decompose(self, sub_tasks: List[Task], actions: Optional[Dict[str, Callable]] = None) -> 'LongTaskPlanner':
        """批量分解目标为子任务序列。"""
        actions = actions or {}
        for task in sub_tasks:
            self.add_step(task, action=actions.get(task.name), retries=1)
        return self

    def _execute_step(self, step: Task) -> Dict[str, Any]:
        """执行单步，优先使用 Orchestrator，否则回退到本地 Worker。"""
        action = self._actions.get(step.id)
        if self.orchestrator is not None:
            record = self.orchestrator.assign_task(step, action=action)
        else:
            record = self.worker.execute_task(step, action=action)
        # 补充计划上下文
        if record is None:
            record = {
                'success': False,
                'result': 'no executor available',
                'duration': 0.0,
                'timestamp': time.time(),
            }
        record['plan_id'] = self.plan_id
        record['step_index'] = self._current_index
        record['task_id'] = step.id
        return record

    def run(self) -> Dict[str, Any]:
        """自驱动执行：循环推进所有子任务直至完成、失败或达到最大步数。"""
        self._status = 'running'
        start = time.time()
        executed = 0

        while self._current_index < len(self._steps) and executed < self.max_steps:
            step = self._steps[self._current_index]
            record = self._execute_step(step)
            executed += 1

            if not record['success']:
                retries_left = self._retry_counts.get(step.id, 0)
                if retries_left > 0:
                    self._retry_counts[step.id] = retries_left - 1
                    continue  # 原地重试当前步骤

            self._results.append(record)

            if not record['success']:
                replanned = False
                if self.replan_callback is not None:
                    new_steps = self.replan_callback(self, self._current_index, record)
                    if new_steps:
                        self._steps = self._steps[:self._current_index + 1] + new_steps
                        # 新步骤默认不重试
                        for t in new_steps:
                            self._retry_counts[t.id] = 0
                        replanned = True
                if not self.continue_on_failure and not replanned:
                    self._status = 'failed'
                    break

            self._current_index += 1

        if self._status != 'failed':
            self._status = 'completed' if self._current_index >= len(self._steps) else 'paused'

        summary = {
            'plan_id': self.plan_id,
            'status': self._status,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'failed_steps': self.failed_steps,
            'duration': time.time() - start,
            'results': self._results,
        }
        return summary

    def checkpoint(self) -> Optional[str]:
        """将当前进度持久化到长期记忆。"""
        if self.memory is None:
            return None
        data = {
            'plan_id': self.plan_id,
            'goal': self.goal.to_dict(),
            'step_index': self._current_index,
            'results': self._results,
            'status': self._status,
            'steps': [step.to_dict() for step in self._steps],
            'timestamp': time.time(),
        }
        return self.memory.save_knowledge(self.plan_id, data)

    def resume(self) -> Optional[Dict[str, Any]]:
        """从长期记忆中的 checkpoint 恢复并继续执行。

        若 checkpoint 损坏或不可读，返回 None 而非抛出异常。
        """
        if self.memory is None:
            return None
        try:
            checkpoint = self.memory.load_knowledge(self.plan_id)
        except Exception as e:
            logger.warning("LongTaskPlanner.resume: failed to load checkpoint %s: %s", self.plan_id, e)
            return None
        if checkpoint is None:
            return None
        self._current_index = checkpoint.get('step_index', 0)
        self._results = checkpoint.get('results', [])
        self._status = 'resumed'
        return self.run()

    def status(self) -> Dict[str, Any]:
        """返回当前计划状态。"""
        return {
            'plan_id': self.plan_id,
            'status': self._status,
            'current_index': self._current_index,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'failed_steps': self.failed_steps,
            'results': self._results,
        }
