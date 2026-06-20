"""自适应编排策略学习。

从历史执行结果中学习不同任务特征下的最优编排策略，使 Agent 能根据
任务的能力需求自动选择成功率最高的策略。

第一性原理：
- 不同任务特征（能力需求组合）对不同策略的响应不同
- 历史执行结果是策略有效性的直接证据
- 任务签名（能力需求集合）是预测策略效果的关键特征
- 探索-利用平衡：未充分尝试的策略应有机会被探索
"""

from typing import Any, Dict, List, Optional, Tuple

from src.task.task_definition import Task

from .orchestrator import MultiAgentOrchestrator


def task_signature(task: Task) -> str:
    """提取任务签名：排序后的能力需求集合，用逗号连接。"""
    caps = task.params.get('required_capabilities', [])
    if not caps:
        return '__generic__'
    return ','.join(sorted(caps))


class StrategyLearner:
    """从历史执行结果中学习最优编排策略。

    维护 (任务签名, 策略) -> (成功数, 总数) 的统计表，
    支持推荐最佳策略与探索-利用平衡。

    :param exploration_rate: 探索概率，未充分采样时优先探索
    :param min_samples: 策略至少需采样此次数才视为已充分评估
    """

    def __init__(
        self,
        exploration_rate: float = 0.1,
        min_samples: int = 3,
    ) -> None:
        self.exploration_rate = max(0.0, min(1.0, exploration_rate))
        self.min_samples = max(1, min_samples)
        self._stats: Dict[Tuple[str, str], Dict[str, int]] = {}

    def record(self, signature: str, strategy: str, success: bool) -> None:
        """记录一次执行结果。"""
        key = (signature, strategy)
        stat = self._stats.setdefault(key, {'successes': 0, 'total': 0})
        stat['total'] += 1
        if success:
            stat['successes'] += 1

    def success_rate(self, signature: str, strategy: str) -> float:
        """查询某签名下某策略的历史成功率。"""
        stat = self._stats.get((signature, strategy))
        if stat is None or stat['total'] == 0:
            return 0.0
        return stat['successes'] / stat['total']

    def recommend(
        self,
        signature: str,
        strategies: List[str],
    ) -> Optional[str]:
        """为任务签名推荐最佳策略。

        优先推荐历史成功率最高且已充分采样的策略；
        若所有策略均未充分采样，返回 None 由调用方默认选择。

        :param signature: 任务签名
        :param strategies: 候选策略列表
        :return: 推荐策略名，或 None
        """
        best_strategy: Optional[str] = None
        best_rate = -1.0
        for strategy in strategies:
            stat = self._stats.get((signature, strategy))
            if stat is None or stat['total'] < self.min_samples:
                continue
            rate = stat['successes'] / stat['total']
            if rate > best_rate:
                best_rate = rate
                best_strategy = strategy
        return best_strategy

    def needs_exploration(self, signature: str, strategies: List[str]) -> Optional[str]:
        """检查是否有策略需要探索（采样不足）。

        :return: 第一个采样不足的策略名，或 None
        """
        for strategy in strategies:
            stat = self._stats.get((signature, strategy))
            if stat is None or stat['total'] < self.min_samples:
                return strategy
        return None

    def get_stats(self) -> List[Dict[str, Any]]:
        """返回全部学习统计。"""
        results: List[Dict[str, Any]] = []
        for (sig, strategy), stat in sorted(self._stats.items()):
            total = stat['total']
            rate = stat['successes'] / total if total > 0 else 0.0
            results.append({
                'signature': sig,
                'strategy': strategy,
                'successes': stat['successes'],
                'total': total,
                'success_rate': rate,
            })
        return results

    def reset(self) -> None:
        """清空学习统计。"""
        self._stats.clear()


class AdaptiveOrchestrator:
    """自适应编排器：包装 MultiAgentOrchestrator，按学习结果自动选择策略。

    :param orchestrator: 被包装的编排器
    :param learner: 策略学习器，默认新建
    :param default_strategy: 无推荐时的默认策略
    """

    def __init__(
        self,
        orchestrator: MultiAgentOrchestrator,
        learner: Optional[StrategyLearner] = None,
        default_strategy: str = 'round_robin',
    ) -> None:
        self.orchestrator = orchestrator
        self.learner = learner or StrategyLearner()
        self.default_strategy = default_strategy
        self._last_strategy: Optional[str] = None

    @property
    def last_strategy(self) -> Optional[str]:
        """返回最近一次自动选择的策略。"""
        return self._last_strategy

    def _select_strategy(self, task: Task) -> str:
        """为任务自动选择策略。"""
        strategies = self.orchestrator.supported_strategies()
        signature = task_signature(task)

        # 探索：有未充分采样的策略时优先探索
        explore = self.learner.needs_exploration(signature, strategies)
        if explore is not None:
            self._last_strategy = explore
            return explore

        # 利用：推荐历史最优
        recommended = self.learner.recommend(signature, strategies)
        chosen = recommended or self.default_strategy
        self._last_strategy = chosen
        return chosen

    def assign_task(
        self,
        task: Task,
        action=None,
        strategy: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """分配任务，自动选择策略并记录学习反馈。"""
        chosen = strategy or self._select_strategy(task)
        self._last_strategy = chosen
        result = self.orchestrator.assign_task(task, action=action, strategy=chosen)
        if result is not None:
            self.learner.record(
                task_signature(task),
                chosen,
                result.get('success', False),
            )
        return result

    def run_mission(
        self,
        tasks: List[Task],
        actions=None,
        strategy: Optional[str] = None,
        mission_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """自驱动执行一组任务，自动选择策略并记录学习反馈。"""
        mission_id = mission_id or f"adaptive_mission_{id(tasks)}"
        results: List[Dict[str, Any]] = []

        for task in tasks:
            action = (actions or {}).get(task.name)
            result = self.assign_task(task, action=action, strategy=strategy)
            if result is not None:
                results.append(result)

        completed = sum(1 for r in results if r.get('success'))
        return {
            'mission_id': mission_id,
            'total_tasks': len(tasks),
            'completed_tasks': completed,
            'failed_tasks': len(tasks) - completed,
            'results': results,
        }
