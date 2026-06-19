import abc
import time
import logging
from enum import Enum, auto
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent运行状态枚举。

    定义Agent在各运行阶段可能处于的状态，包括空闲、正常运行、复盘、紧急、维护和关闭。
    """
    IDLE = auto()          # 空闲等待
    ACTIVE = auto()        # 正常运行
    REVIEW = auto()        # 离线复盘
    EMERGENCY = auto()     # 紧急模式
    MAINTENANCE = auto()   # 维护模式
    SHUTDOWN = auto()      # 关闭中


@dataclass
class StateTransition:
    """状态转换记录。

    记录一次状态转换的源状态、目标状态、原因和时间戳。
    """
    from_state: AgentState
    to_state: AgentState
    reason: str
    timestamp: float = field(default_factory=time.time)


class BaseStateMachine(abc.ABC):
    """Agent状态机抽象基类：管理运行模式切换。

    提供状态转换验证、处理器注册和状态查询的抽象接口。
    """

    # 合法状态转换图
    VALID_TRANSITIONS: Dict[AgentState, List[AgentState]] = {
        AgentState.IDLE: [AgentState.ACTIVE, AgentState.MAINTENANCE, AgentState.SHUTDOWN],
        AgentState.ACTIVE: [AgentState.IDLE, AgentState.REVIEW, AgentState.EMERGENCY,
                           AgentState.MAINTENANCE, AgentState.SHUTDOWN],
        AgentState.REVIEW: [AgentState.ACTIVE, AgentState.IDLE, AgentState.EMERGENCY],
        AgentState.EMERGENCY: [AgentState.ACTIVE, AgentState.MAINTENANCE, AgentState.SHUTDOWN],
        AgentState.MAINTENANCE: [AgentState.IDLE, AgentState.SHUTDOWN],
        AgentState.SHUTDOWN: [],
    }

    def __init__(self, initial_state: AgentState = AgentState.IDLE) -> None:
        """初始化状态机。

        :param initial_state: 初始状态。
        :type initial_state: AgentState
        :return: 无。
        :rtype: None
        """
        self._state = initial_state
        self._previous_state: Optional[AgentState] = None
        self._transition_history: List[StateTransition] = []
        self._state_entry_time: float = time.time()
        self._init_handlers()

    def _init_handlers(self) -> None:
        self._handlers: Dict[AgentState, List[Callable[[AgentState, AgentState], None]]] = {
            state: [] for state in AgentState
        }
        self._transition_handlers: List[Callable[[StateTransition], None]] = []

    @property
    def state(self) -> AgentState:
        """当前状态。

        :param: 无
        :type: 无
        :return: 当前 AgentState。
        :rtype: AgentState
        """
        return self._state

    @property
    def previous_state(self) -> Optional[AgentState]:
        """上一个状态。

        :param: 无
        :type: 无
        :return: 上一个 AgentState 或 None。
        :rtype: Optional[AgentState]
        """
        return self._previous_state

    @abc.abstractmethod
    def can_transition(self, target: AgentState) -> bool:
        """检查是否可以转换到目标状态。

        :param target: 目标状态。
        :type target: AgentState
        :return: 是否允许转换。
        :rtype: bool
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def transition(self, target: AgentState, reason: str = "") -> bool:
        """执行状态转换。

        :param target: 目标状态。
        :type target: AgentState
        :param reason: 转换原因。
        :type reason: str
        :return: 转换是否成功。
        :rtype: bool
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def on_enter(self, state: AgentState,
                 handler: Callable[[AgentState, AgentState], None]) -> None:
        """注册进入某状态的处理器。

        :param state: 目标状态。
        :type state: AgentState
        :param handler: 处理函数。
        :type handler: Callable[[AgentState, AgentState], None]
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def on_transition(self, handler: Callable[[StateTransition], None]) -> None:
        """注册通用转换处理器。

        :param handler: 处理函数。
        :type handler: Callable[[StateTransition], None]
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def get_state_duration(self) -> float:
        """获取当前状态持续时间。

        :return: 持续时间（秒）。
        :rtype: float
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """获取最近的转换历史。

        :param limit: 返回条数。
        :type limit: int
        :return: 转换记录列表。
        :rtype: List[StateTransition]
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def is_operational(self) -> bool:
        """Agent是否处于可操作状态。

        :return: 是否可操作。
        :rtype: bool
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def is_critical(self) -> bool:
        """Agent是否处于紧急状态。

        :return: 是否紧急。
        :rtype: bool
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def auto_transition(self, safety_result: Dict[str, Any],
                        review_ready: bool = False,
                        maintenance_requested: bool = False) -> Optional[StateTransition]:
        """根据条件自动判断并执行状态转换。

        :param safety_result: 安全检查结果。
        :type safety_result: Dict[str, Any]
        :param review_ready: 是否准备好复盘。
        :type review_ready: bool
        :param maintenance_requested: 是否请求维护。
        :type maintenance_requested: bool
        :return: 转换记录或 None。
        :rtype: Optional[StateTransition]
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取状态统计。

        :return: 统计字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        ...


class StateMachine(BaseStateMachine):
    """Agent状态机：管理运行模式切换。

    实现基于合法状态转换图的状态管理，支持手动转换、自动转换、处理器注册和状态统计。
    """

    def __init__(self, initial_state: AgentState = AgentState.IDLE) -> None:
        """初始化状态机。

        :param initial_state: 初始状态。
        :type initial_state: AgentState
        :return: 无。
        :rtype: None
        """
        super().__init__(initial_state)

    def can_transition(self, target: AgentState) -> bool:
        """检查是否可以转换到目标状态。

        :param target: 目标状态。
        :type target: AgentState
        :return: 是否允许转换。
        :rtype: bool
        :raises: 无。
        """
        return target in self.VALID_TRANSITIONS.get(self._state, [])

    def transition(self, target: AgentState, reason: str = "") -> bool:
        """执行状态转换。

        :param target: 目标状态。
        :type target: AgentState
        :param reason: 转换原因。
        :type reason: str
        :return: 转换是否成功。
        :rtype: bool
        :raises: 无。
        """
        if not self.can_transition(target):
            self._log_invalid_transition(target)
            return False
        self._perform_transition(target, reason)
        return True

    def _perform_transition(self, target: AgentState, reason: str) -> None:
        self._apply_state_change(target)
        transition = self._record_transition(target, reason)
        self._log_transition(transition, reason)
        self._trigger_handlers(target, transition)
        self._trigger_transition_handlers(transition)

    def _log_invalid_transition(self, target: AgentState) -> None:
        """记录无效转换警告。"""
        logger.warning(
            f"Invalid transition: {self._state.name} -> {target.name}"
        )

    def _apply_state_change(self, target: AgentState) -> None:
        """应用状态变更。"""
        self._previous_state = self._state
        self._state = target
        self._state_entry_time = time.time()

    def _record_transition(self, target: AgentState, reason: str) -> StateTransition:
        """记录状态转换。"""
        transition = StateTransition(
            from_state=self._previous_state,
            to_state=target,
            reason=reason or "manual"
        )
        self._transition_history.append(transition)
        return transition

    def _log_transition(self, transition: StateTransition, reason: str) -> None:
        """记录状态转换日志。"""
        logger.info(
            f"State transition: {transition.from_state.name} -> "
            f"{transition.to_state.name} ({reason})"
        )

    def _trigger_handlers(self, target: AgentState, transition: StateTransition) -> None:
        """触发状态处理器。

        :param target: 目标状态。
        :type target: AgentState
        :param transition: 转换记录。
        :type transition: StateTransition
        """
        for handler in self._handlers.get(target, []):
            self._safe_call_handler(handler, transition.from_state, target)

    def _safe_call_handler(self, handler: Callable[[AgentState, AgentState], None], from_state: AgentState, target: AgentState) -> None:
        try:
            handler(from_state, target)
        except Exception as e:
            logger.error(f"State handler error: {e}")

    def _trigger_transition_handlers(self, transition: StateTransition) -> None:
        """触发通用转换处理器。

        :param transition: 转换记录。
        :type transition: StateTransition
        """
        for handler in self._transition_handlers:
            self._safe_call_transition_handler(handler, transition)

    def _safe_call_transition_handler(self, handler: Callable[[StateTransition], None], transition: StateTransition) -> None:
        try:
            handler(transition)
        except Exception as e:
            logger.error(f"Transition handler error: {e}")

    def on_enter(self, state: AgentState,
                 handler: Callable[[AgentState, AgentState], None]) -> None:
        """注册进入某状态的处理器。

        :param state: 目标状态。
        :type state: AgentState
        :param handler: 处理函数。
        :type handler: Callable[[AgentState, AgentState], None]
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        self._handlers[state].append(handler)

    def on_transition(self, handler: Callable[[StateTransition], None]) -> None:
        """注册通用转换处理器。

        :param handler: 处理函数。
        :type handler: Callable[[StateTransition], None]
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        self._transition_handlers.append(handler)

    def get_state_duration(self) -> float:
        """获取当前状态持续时间。

        :return: 持续时间（秒）。
        :rtype: float
        :raises: 无。
        """
        return time.time() - self._state_entry_time

    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """获取最近的转换历史。

        :param limit: 返回条数。
        :type limit: int
        :return: 转换记录列表。
        :rtype: List[StateTransition]
        :raises: 无。
        """
        return self._transition_history[-limit:]

    def is_operational(self) -> bool:
        """Agent是否处于可操作状态。

        :return: 是否可操作。
        :rtype: bool
        :raises: 无。
        """
        return self._state in (AgentState.IDLE, AgentState.ACTIVE, AgentState.REVIEW)

    def is_critical(self) -> bool:
        """Agent是否处于紧急状态。

        :return: 是否紧急。
        :rtype: bool
        :raises: 无。
        """
        return self._state == AgentState.EMERGENCY

    def auto_transition(self, safety_result: Dict[str, Any],
                        review_ready: bool = False,
                        maintenance_requested: bool = False) -> Optional[StateTransition]:
        """根据条件自动判断并执行状态转换。

        :param safety_result: 安全检查结果。
        :type safety_result: Dict[str, Any]
        :param review_ready: 是否准备好复盘。
        :type review_ready: bool
        :param maintenance_requested: 是否请求维护。
        :type maintenance_requested: bool
        :return: 转换记录或 None。
        :rtype: Optional[StateTransition]
        :raises: 无。
        """
        current = self._state
        return self._try_all_transitions(safety_result, review_ready, maintenance_requested, current)

    def _try_all_transitions(self, safety_result: Dict[str, Any], review_ready: bool,
                             maintenance_requested: bool, current: AgentState) -> Optional[StateTransition]:
        return self._try_safety_transitions(safety_result, current) or self._try_review_transitions(review_ready, current) or self._try_maintenance_transitions(maintenance_requested, current) or self._try_idle_activation(current)

    def _try_safety_transitions(self, safety_result: Dict[str, Any], current: AgentState) -> Optional[StateTransition]:
        """尝试安全相关转换。"""
        if self._check_emergency(safety_result, current):
            return self._transition_history[-1]
        if self._check_recovery(safety_result, current):
            return self._transition_history[-1]
        return None

    def _try_review_transitions(self, review_ready: bool, current: AgentState) -> Optional[StateTransition]:
        """尝试复盘相关转换。"""
        if self._check_review(review_ready, current):
            return self._transition_history[-1]
        if self._check_review_complete(review_ready, current):
            return self._transition_history[-1]
        return None

    def _try_maintenance_transitions(self, maintenance_requested: bool, current: AgentState) -> Optional[StateTransition]:
        """尝试维护相关转换。"""
        if self._check_maintenance(maintenance_requested, current):
            return self._transition_history[-1]
        return None

    def _try_idle_activation(self, current: AgentState) -> Optional[StateTransition]:
        """尝试空闲激活转换。"""
        if self._check_idle_activation(current):
            return self._transition_history[-1]
        return None

    def _check_emergency(self, safety_result: Dict[str, Any], current: AgentState) -> bool:
        """检查紧急模式触发。"""
        if safety_result.get('emergency_mode', False):
            if current != AgentState.EMERGENCY:
                self.transition(AgentState.EMERGENCY, "safety_violation")
                return True
        return False

    def _check_recovery(self, safety_result: Dict[str, Any], current: AgentState) -> bool:
        """检查从紧急恢复。"""
        if current == AgentState.EMERGENCY and not safety_result.get('emergency_mode', False):
            self.transition(AgentState.ACTIVE, "safety_cleared")
            return True
        return False

    def _check_review(self, review_ready: bool, current: AgentState) -> bool:
        """检查复盘触发。"""
        if review_ready and current == AgentState.ACTIVE:
            self.transition(AgentState.REVIEW, "review_scheduled")
            return True
        return False

    def _check_review_complete(self, review_ready: bool, current: AgentState) -> bool:
        """检查复盘完成。"""
        if current == AgentState.REVIEW and not review_ready:
            self.transition(AgentState.ACTIVE, "review_completed")
            return True
        return False

    def _check_maintenance(self, maintenance_requested: bool, current: AgentState) -> bool:
        """检查维护请求。"""
        if maintenance_requested and current in (AgentState.IDLE, AgentState.ACTIVE):
            self.transition(AgentState.MAINTENANCE, "maintenance_requested")
            return True
        return False

    def _check_idle_activation(self, current: AgentState) -> bool:
        """检查空闲激活。"""
        if current == AgentState.IDLE:
            self.transition(AgentState.ACTIVE, "activation")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取状态统计。

        :return: 统计字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        return self._build_stats_dict()

    def _build_stats_dict(self) -> Dict[str, Any]:
        return dict(current_state=self._state.name, previous_state=self._previous_state.name if self._previous_state else None, state_duration_seconds=round(self.get_state_duration(), 2), total_transitions=len(self._transition_history), is_operational=self.is_operational(), is_critical=self.is_critical())
