import abc
import time
from typing import Dict, Any, List


class BaseValueScorer(abc.ABC):
    """Abstract base class for value scoring.

    :param error_weight: 误差权重
    :type error_weight: float
    :param novelty_weight: 新颖性权重
    :type novelty_weight: float
    :param decay_hours: 衰减小时数
    :type decay_hours: float
    :return: 无
    :rtype: None
    """

    def __init__(self, error_weight: float = 0.6, novelty_weight: float = 0.4,
                 decay_hours: float = 168.0) -> None:
        """Initialize the base value scorer.

        :param error_weight: 误差权重
        :type error_weight: float
        :param novelty_weight: 新颖性权重
        :type novelty_weight: float
        :param decay_hours: 衰减小时数
        :type decay_hours: float
        :return: 无
        :rtype: None
        """
        self.error_weight = error_weight
        self.novelty_weight = novelty_weight
        self.decay_hours = decay_hours

    @abc.abstractmethod
    def _abstract_score_op(self) -> None:
        """Abstract scoring operation placeholder.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """

    def compute_score(self, sample: Dict[str, Any]) -> float:
        """Compute the value score for a sample.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 价值分数
        :rtype: float
        """
        mae = sample.get('mae', 0.0)
        error_value = min(mae, 1.0)
        novelty_value = self._compute_novelty(sample)
        age_hours = (time.time() - sample.get('timestamp', time.time())) / 3600
        time_decay = max(0.1, 1.0 - age_hours / self.decay_hours)
        score = (error_value * self.error_weight +
                 novelty_value * self.novelty_weight) * time_decay
        return round(max(0.0, min(1.0, score)), 4)

    def _compute_novelty(self, sample: Dict[str, Any]) -> float:
        """Compute novelty component for a sample.

        :param sample: 样本数据
        :type sample: Dict[str, Any]
        :return: 新颖性得分
        :rtype: float
        """
        task_id = sample.get('task_id')
        if task_id is None:
            return 0.5
        errors = sample.get('errors', {})
        if not errors:
            return 0.5
        max_error = max(errors.values()) if errors else 0.0
        return min(1.0, max_error)

    def batch_score(self, samples: List[Dict[str, Any]]) -> Dict[int, float]:
        """Compute scores for a batch of samples.

        :param samples: 样本列表
        :type samples: List[Dict[str, Any]]
        :return: 样本 ID 到得分的映射
        :rtype: Dict[int, float]
        """
        scores = {}
        for sample in samples:
            sample_id = sample.get('id')
            if sample_id is not None:
                scores[sample_id] = self.compute_score(sample)
        return scores


class ValueScorer(BaseValueScorer):
    """Concrete value scorer implementation.

    :param: 无
    :type: 无
    :return: 无
    :rtype: None
    """

    def _abstract_score_op(self) -> None:
        """Concrete implementation of abstract scoring operation.

        :param: 无
        :type: 无
        :return: 无
        :rtype: None
        """
        pass
