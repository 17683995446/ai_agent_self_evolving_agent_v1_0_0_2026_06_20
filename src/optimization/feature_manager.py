import abc
from typing import Dict, Any, List, Set
from collections import deque


class BaseFeatureManager(abc.ABC):
    def __init__(self, base_features: List[str]) -> None:
        """Initialize the feature manager.

        :param base_features: List of base feature names.
        :type base_features: List[str]
        :return: None
        :rtype: None
        """
        self.active_features: Set[str] = set(base_features)
        self.candidate_features: Set[str] = set()
        self._feature_scores: Dict[str, deque] = {}
        self._min_samples_for_eval: int = 10

    @abc.abstractmethod
    def _abstract_feature_op(self) -> None:
        """Abstract feature operation placeholder."""

    def add_candidate(self, feature: str) -> None:
        """Add a candidate feature.

        :param feature: Candidate feature name.
        :type feature: str
        :return: None
        :rtype: None
        """
        if feature not in self.active_features:
            self.candidate_features.add(feature)
            self._feature_scores[feature] = deque(maxlen=self._min_samples_for_eval)

    def evaluate_and_promote(self, feature_errors: Dict[str, float]) -> List[str]:
        """Evaluate candidates and promote good ones.

        :param feature_errors: Mapping of feature names to error values.
        :type feature_errors: Dict[str, float]
        :return: List of promoted feature names.
        :rtype: List[str]
        """
        promoted = []
        for feature, error in feature_errors.items():
            if self._should_promote(feature, error):
                promoted.append(feature)
        return promoted

    def _should_promote(self, feature: str, error: float) -> bool:
        if feature not in self.candidate_features:
            return False
        self._feature_scores[feature].append(error)
        if len(self._feature_scores[feature]) < self._min_samples_for_eval:
            return False
        avg_error = sum(self._feature_scores[feature]) / len(self._feature_scores[feature])
        if avg_error >= 0.3:
            return False
        self.active_features.add(feature)
        self.candidate_features.discard(feature)
        return True

    def demote_underperforming(self, feature_errors: Dict[str, float],
                                threshold: float = 0.8) -> List[str]:
        """Demote active features with high error.

        :param feature_errors: Mapping of feature names to error values.
        :type feature_errors: Dict[str, float]
        :param threshold: Error threshold for demotion.
        :type threshold: float
        :return: List of demoted feature names.
        :rtype: List[str]
        """
        demoted = []
        for feature, error in feature_errors.items():
            if feature in self.active_features and error > threshold:
                self.active_features.discard(feature)
                demoted.append(feature)
        return demoted

    def get_active_features(self) -> List[str]:
        """Return sorted list of active features.

        :param: 无
        :type: 无
        :return: Sorted list of active feature names.
        :rtype: List[str]
        """
        return sorted(self.active_features)

    def get_candidate_features(self) -> List[str]:
        """Return sorted list of candidate features.

        :param: 无
        :type: 无
        :return: Sorted list of candidate feature names.
        :rtype: List[str]
        """
        return sorted(self.candidate_features)


class FeatureManager(BaseFeatureManager):
    def _abstract_feature_op(self) -> None:
        """Concrete implementation of abstract feature operation."""
        pass


class BaseModelReassembler(abc.ABC):
    def __init__(self, predictor_registry: Dict[str, Any]) -> None:
        """Initialize the model reassembler.

        :param predictor_registry: Mapping of feature names to predictor instances.
        :type predictor_registry: Dict[str, Any]
        :return: None
        :rtype: None
        """
        self.registry = predictor_registry
        self._assembly_history: deque = deque(maxlen=10)

    @abc.abstractmethod
    def _abstract_assemble_op(self) -> None:
        """Abstract assembly operation placeholder."""

    def reassemble(self, active_features: List[str]) -> Dict[str, Any]:
        """Reassemble predictors for active features.

        :param active_features: List of active feature names.
        :type active_features: List[str]
        :return: Mapping of feature names to predictor instances.
        :rtype: Dict[str, Any]
        """
        new_assembly = {}
        for feature in active_features:
            new_assembly[feature] = self._get_predictor(feature)
        self._record_assembly(active_features)
        return new_assembly

    def _get_predictor(self, feature: str) -> Any:
        if feature in self.registry:
            return self.registry[feature]
        from src.prediction.base_predictor import AdaptiveExponentialSmoother
        return AdaptiveExponentialSmoother()

    def _record_assembly(self, active_features: List[str]) -> None:
        self._assembly_history.append({
            'features': list(active_features),
            'count': len(active_features),
        })

    def get_assembly_stats(self) -> Dict[str, Any]:
        """Return assembly history statistics.

        :param: 无
        :type: 无
        :return: Dictionary containing assembly history statistics.
        :rtype: Dict[str, Any]
        """
        return {
            'history_count': len(self._assembly_history),
            'latest_feature_count': self._assembly_history[-1]['count'] if self._assembly_history else 0,
        }


class ModelReassembler(BaseModelReassembler):
    def _abstract_assemble_op(self) -> None:
        """Concrete implementation of abstract assembly operation."""
        pass
