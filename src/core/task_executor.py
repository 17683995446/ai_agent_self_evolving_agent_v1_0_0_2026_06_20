"""任务执行器组件。

从 agent_loop.py 提取的任务执行逻辑，遵循单一职责原则：
将任务合法性校验、动作分发、执行后状态更新等关注点从主循环中分离。

:author: self_evolving_agent
:since: 阶段158
"""

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class TaskExecutor:
    """任务执行器。

    负责单个任务的完整执行流程：合法性校验→动作分发→执行→状态更新。
    通过依赖注入接收所需组件，不直接持有全局状态。

    :param behavior_guardian: 行为护栏，用于路径校验与探索频率控制。
    :param task_pool: 任务池，用于标记执行结果与冻结异常任务。
    :param state_vector: 状态向量，用于更新任务成功率统计。
    """

    def __init__(self, behavior_guardian: Any, task_pool: Any, state_vector: Any) -> None:
        self.behavior_guardian = behavior_guardian
        self.task_pool = task_pool
        self.state_vector = state_vector

    def execute(self, task: Any, state: Dict[str, Any],
                high_preds: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务并返回结果。

        :param task: 任务对象，为None时返回no_task状态。
        :type task: Any
        :param state: 当前状态。
        :type state: Dict[str, Any]
        :param high_preds: 高层预测结果。
        :type high_preds: Dict[str, Any]
        :return: 执行结果字典。
        :rtype: Dict[str, Any]
        """
        if task is None:
            return {'status': 'no_task', 'success': True}
        if self._is_forbidden(task):
            return {'status': 'forbidden', 'success': False}
        return self._execute_valid_task(task, state, high_preds)

    def _execute_valid_task(self, task: Any, state: Dict[str, Any],
                            high_preds: Dict[str, Any]) -> Dict[str, Any]:
        """执行合法任务并更新各组件状态。"""
        success, reward = self._run_task_action(task, state, high_preds)
        self._update_task_after_execution(task, success)
        self.task_pool.mark_executed(task.id, success, reward)
        self.state_vector.update_task_success(success)
        return {'status': 'executed', 'task_id': task.id, 'success': success, 'reward': reward}

    def _update_task_after_execution(self, task: Any, success: bool) -> None:
        """执行后检查连续失败次数，必要时冻结任务。"""
        if not self.behavior_guardian.check_exploration(task.id, success):
            logger.warning("Task %s frozen due to consecutive failures", task.id)
            self.task_pool.freeze_task(task.id)

    def _is_forbidden(self, task: Any) -> bool:
        """检查任务是否触碰禁止路径。"""
        if hasattr(task, 'target_path') and task.target_path:
            if not self.behavior_guardian.check_path(task.target_path):
                logger.warning("Task %s forbidden path: %s", task.id, task.target_path)
                self.task_pool.freeze_task(task.id)
                return True
        return False

    def _run_task_action(self, task: Any, state: Dict[str, Any],
                         high_preds: Dict[str, Any]) -> Tuple[bool, float]:
        """运行任务动作并返回成功标志与奖励值。"""
        try:
            success = self._execute_action(task, state, high_preds)
            return success, 1.0 if success else 0.0
        except Exception as e:
            logger.error("Task execution error: %s", e)
            return False, 0.0

    def _execute_action(self, task: Any, state: Dict[str, Any],
                        high_preds: Dict[str, Any]) -> bool:
        """根据高层预测推荐动作分发执行。"""
        action = high_preds.get('recommended_action', 'continue_exploration')
        if action == 'emergency_safeguard':
            return self._do_safeguard(task, state)
        elif action == 'increase_monitoring':
            return self._do_monitoring(task, state)
        return self._do_exploration(task, state)

    def _do_safeguard(self, task: Any, state: Dict[str, Any]) -> bool:
        """执行保护动作。"""
        logger.info("Safeguard action for task %s", task.id)
        return True

    def _do_monitoring(self, task: Any, state: Dict[str, Any]) -> bool:
        """执行监控动作。"""
        logger.info("Monitoring action for task %s", task.id)
        return True

    def _do_exploration(self, task: Any, state: Dict[str, Any]) -> bool:
        """执行探索动作。"""
        logger.info("Exploration action for task %s: %s", task.id, task.name)
        return True
