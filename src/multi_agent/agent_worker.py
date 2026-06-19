import time
import uuid
from typing import Dict, Any, List, Optional, Callable

from src.task.task_pool import TaskPool
from src.task.task_definition import Task


class AgentWorker:
    """轻量级 Agent 工作节点，可独立执行任务并维护本地任务池。

    :param worker_id: 工作节点唯一标识
    :param capabilities: 能力标签集合，用于编排器做能力匹配
    :param task_pool: 可选的任务池实例
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        task_pool: Optional[TaskPool] = None,
    ) -> None:
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.capabilities: List[str] = list(capabilities) if capabilities else []
        self.task_pool = task_pool or TaskPool()
        self._execution_log: List[Dict[str, Any]] = []
        self._busy = False

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

    def is_busy(self) -> bool:
        """当前是否正在执行任务。"""
        return self._busy

    def get_stats(self) -> Dict[str, Any]:
        """返回 worker 统计信息。"""
        total = len(self._execution_log)
        success_count = sum(1 for r in self._execution_log if r['success'])
        return {
            'worker_id': self.worker_id,
            'capabilities': self.capabilities,
            'total_executions': total,
            'success_executions': success_count,
            'success_rate': success_count / total if total > 0 else 0.0,
            'busy': self._busy,
            'task_pool_size': len(self.task_pool.tasks),
        }
