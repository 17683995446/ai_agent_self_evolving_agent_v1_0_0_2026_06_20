import abc
import time
from typing import Dict, Any, List, Optional

from .short_term import ShortTermMemory
from .value_scorer import ValueScorer
from ..optimization.micro_optimizer import MicroOptimizer


class BaseReviewEngine(abc.ABC):
    """Abstract base class for review engines.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    @abc.abstractmethod
    def run_review(self) -> Dict[str, Any]:
        """Run the review process and return results.

        :param: 无
        :type: 无
        :return: 审查结果
        :rtype: Dict[str, Any]
        """
        pass

    @abc.abstractmethod
    def should_review(self, sample_count: int, idle_cpu: float = 0.0) -> bool:
        """Determine whether a review should be run.

        :param sample_count: 样本数量
        :type sample_count: int
        :param idle_cpu: 空闲 CPU 百分比
        :type idle_cpu: float
        :return: 是否应该运行审查
        :rtype: bool
        """
        pass


class ReviewEngine(BaseReviewEngine):
    """Concrete review engine implementation.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    def __init__(self, memory: ShortTermMemory, optimizer: MicroOptimizer,
                 min_samples: int = 100) -> None:
        """Initialize the review engine.

        :param memory: 短期记忆实例
        :type memory: ShortTermMemory
        :param optimizer: 微优化器实例
        :type optimizer: MicroOptimizer
        :param min_samples: 最小样本数量
        :type min_samples: int
        :return: 无
        :rtype: None
        """
        self.memory = memory
        self.optimizer = optimizer
        self.min_samples = min_samples
        self.value_scorer = ValueScorer()

    def should_review(self, sample_count: int, idle_cpu: float = 0.0) -> bool:
        """Determine whether a review should be run.

        :param sample_count: 样本数量
        :type sample_count: int
        :param idle_cpu: 空闲 CPU 百分比
        :type idle_cpu: float
        :return: 是否应该运行审查
        :rtype: bool
        """
        if sample_count < self.min_samples:
            return False
        if idle_cpu < 10.0:
            return False
        return True

    def run_review(self) -> Dict[str, Any]:
        """Run the review process and return results.

        :param: 无
        :type: 无
        :return: 审查结果
        :rtype: Dict[str, Any]
        """
        start_time = time.time()
        samples = self.memory.load_high_value_samples(limit=1000)
        if not samples:
            return {'status': 'no_samples', 'processed': 0}
        self._batch_update_value_scores(samples)
        replayed = self._replay_samples(samples)
        self.memory.cleanup_old_samples(retention_ratio=0.3)
        duration = time.time() - start_time
        return {
            'status': 'completed',
            'processed': len(samples),
            'replayed': replayed,
            'duration_seconds': round(duration, 2),
            'timestamp': time.time(),
        }

    def _batch_update_value_scores(self, samples: List[Dict[str, Any]]) -> int:
        """Update value scores for a batch of samples.

        :param samples: 样本列表
        :type samples: List[Dict[str, Any]]
        :return: 更新的样本数量
        :rtype: int
        """
        updated = 0
        for sample in samples:
            score = self.value_scorer.compute_score(sample)
            sample_id = sample.get('id')
            if sample_id is not None:
                self.memory.update_value_score(sample_id, score)
                updated += 1
        return updated

    def _replay_samples(self, samples: List[Dict[str, Any]]) -> int:
        """Replay samples and record errors.

        :param samples: 样本列表
        :type samples: List[Dict[str, Any]]
        :return: 重放的样本数量
        :rtype: int
        """
        replayed = 0
        for sample in samples:
            errors = sample.get('error_json', {})
            if isinstance(errors, dict) and errors:
                self.optimizer.record_errors(errors)
                replayed += 1
        return replayed

    def generate_report(self, review_result: Dict[str, Any]) -> str:
        """Generate a text report from review results.

        :param review_result: 审查结果
        :type review_result: Dict[str, Any]
        :return: 文本报告
        :rtype: str
        """
        lines = [
            "=" * 40,
            "Offline Review Report",
            "=" * 40,
            f"Status: {review_result.get('status', 'unknown')}",
            f"Samples Processed: {review_result.get('processed', 0)}",
            f"Samples Replayed: {review_result.get('replayed', 0)}",
            f"Duration: {review_result.get('duration_seconds', 0):.2f}s",
            "=" * 40,
        ]
        return "\n".join(lines)
