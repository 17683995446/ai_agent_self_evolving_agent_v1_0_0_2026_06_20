import abc
import os
import sys
import time
import logging
import logging.handlers
import threading
from typing import Dict, Any, Optional, Tuple
from collections import deque

from src.core.config import Config
from src.core.constants import DEFAULT_CONFIG_PATH
from src.perception.state_vector import StateVector
from src.prediction.layered_predictor import LowLevelPredictor, MidLevelPredictor, HighLevelPredictor
from src.prediction.error_evaluator import ErrorEvaluator
from src.drive.reward_engine import RewardEngine
from src.task.task_pool import TaskPool
from src.task.task_generator import TaskGenerator
from src.memory.short_term import ShortTermMemory
from src.memory.review_engine import ReviewEngine
from src.optimization.micro_optimizer import MicroOptimizer
from src.optimization.snapshot_manager import SnapshotManager
from src.optimization.auto_rollback import AutoRollback
from src.safety.guardian import ResourceGuardian, BehaviorGuardian, SafetyEnforcer
from src.dashboard.server import DashboardServer
from src.monitoring.alert_manager import AlertManager


logger = logging.getLogger(__name__)


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """安全的日志文件轮转处理器。

    在磁盘满、IO错误等场景下捕获OSError，避免日志写入异常导致应用崩溃，
    同时向标准错误输出简要提示。
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except OSError as e:
            try:
                sys.stderr.write(f"[LOGGING ERROR] failed to write log: {e}\n")
            except OSError:
                pass

    def handleError(self, record: logging.LogRecord) -> None:
        """覆盖默认错误处理，防止异常上抛。"""
        try:
            sys.stderr.write(f"[LOGGING ERROR] handler error for record {record.name}\n")
        except OSError:
            pass


class BaseAgentLoop(abc.ABC):
    """核心Agent主循环抽象基类。

    定义Agent主循环的通用初始化接口和抽象方法，包括运行、停止和统计信息获取。
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        """初始化Agent主循环。

        :param config_path: 配置文件路径。
        :type config_path: str
        :return: 无。
        :rtype: None
        """
        self._load_config(config_path)
        self._setup_logging()
        self._init_all_subsystems()

    def _load_config(self, config_path: str) -> None:
        self.config = Config(config_path)
        self.cfg = self.config.to_dict()

    def _init_all_subsystems(self) -> None:
        self._init_state_vector()
        self._init_predictors()
        self._init_task_system()
        self._init_memory()
        self._init_optimization()
        self._init_safety()
        self._init_dashboard()
        self._init_alert_manager()
        self._init_counters()

    @abc.abstractmethod
    def run_cycle(self) -> Dict[str, Any]:
        """执行一个完整的主循环周期。

        :return: 周期结果字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def run(self, max_cycles: Optional[int] = None,
            sleep_interval: float = 1.0) -> None:
        """持续运行主循环。

        :param max_cycles: 最大周期数。
        :type max_cycles: Optional[int]
        :param sleep_interval: 周期间隔（秒）。
        :type sleep_interval: float
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """停止主循环。

        :return: 无。
        :rtype: None
        :raises: 无。
        """
        ...

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        :return: 统计字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        ...


class AgentLoop(BaseAgentLoop):
    """核心Agent主循环：感知→预测→驱力→任务→执行→修正→记忆→复盘。

    实现完整的Agent运行周期，包括状态感知、多层预测、安全检测、任务生成与执行、
    记忆存储、参数优化、复盘和快照管理等步骤。
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        """初始化Agent主循环。

        :param config_path: 配置文件路径。
        :type config_path: str
        :return: 无。
        :rtype: None
        """
        super().__init__(config_path)

    def _init_state_vector(self) -> None:
        """初始化状态向量与奖励引擎。"""
        self.state_vector = StateVector(self.cfg)
        self.error_evaluator = ErrorEvaluator()
        self._init_reward_engine()

    def _init_reward_engine(self) -> None:
        _bw = self.cfg.get('base_weights', {})
        base_weights = (
            _bw.get('safe', 0.4),
            _bw.get('gain', 0.4),
            _bw.get('eff', 0.2),
        )
        self.reward_engine = RewardEngine(base_weights=base_weights)

    def _init_predictors(self) -> None:
        """初始化多层预测器。"""
        metrics = ['cpu_percent', 'mem_percent', 'disk_percent']
        self.low_predictor = LowLevelPredictor(metrics=metrics)
        self.mid_predictor = MidLevelPredictor(metrics=metrics)
        self.high_predictor = HighLevelPredictor()

    def _init_task_system(self) -> None:
        """初始化任务池与生成器。"""
        task_pool_cfg = self.cfg.get('task_pool', {})
        max_size = task_pool_cfg.get('max_size', 50)
        if not isinstance(max_size, int) or max_size <= 0:
            max_size = 50
        self.task_pool = TaskPool(max_size=max_size)
        self.task_generator = TaskGenerator()

    def _init_memory(self) -> None:
        """初始化短期记忆。"""
        data_dir = self.cfg.get('paths', {}).get('data_dir', 'data')
        os.makedirs(data_dir, exist_ok=True)
        self.memory = ShortTermMemory(
            db_path=os.path.join(data_dir, "memory.db"),
            max_samples=self.cfg.get('memory', {}).get('max_short_term_samples', 10000)
        )

    def _init_optimization(self) -> None:
        """初始化优化器、复盘引擎与快照管理。"""
        self.optimizer = MicroOptimizer()
        self._init_review_engine()
        self._init_snapshot_manager()

    def _init_review_engine(self) -> None:
        self.review_engine = ReviewEngine(
            memory=self.memory,
            optimizer=self.optimizer,
            min_samples=self.cfg.get('review', {}).get('min_samples', 100)
        )

    def _init_snapshot_manager(self) -> None:
        snapshot_dir = self.cfg.get('paths', {}).get('snapshot_dir', 'snapshots')
        self.snapshot_manager = SnapshotManager(snapshot_dir=snapshot_dir)
        ar_cfg = self.cfg.get('auto_rollback', {})
        degradation_threshold = ar_cfg.get('mae_degradation_threshold', 0.2)
        if not isinstance(degradation_threshold, (int, float)) or degradation_threshold < 0:
            degradation_threshold = 0.2
        self.auto_rollback = AutoRollback(
            snapshot_manager=self.snapshot_manager,
            mae_degradation_threshold=degradation_threshold
        )

    def _init_safety(self) -> None:
        """初始化安全护栏。"""
        self._init_resource_guardian()
        self._init_behavior_guardian()
        self.safety_enforcer = SafetyEnforcer(
            self.resource_guardian,
            self.behavior_guardian
        )

    def _init_resource_guardian(self) -> None:
        self.resource_guardian = ResourceGuardian(
            max_cpu_percent=self.cfg.get('resource_limits', {}).get('max_cpu_percent', 20.0),
            max_mem_percent=self.cfg.get('resource_limits', {}).get('max_mem_percent', 20.0),
        )

    def _init_behavior_guardian(self) -> None:
        self.behavior_guardian = BehaviorGuardian(
            forbidden_paths=self.cfg.get('safety', {}).get('forbidden_paths'),
            max_consecutive_failures=self.cfg.get('safety', {}).get('max_consecutive_failures', 5)
        )

    def _init_dashboard(self) -> None:
        """初始化仪表盘（可选）。"""
        self.dashboard: Optional[DashboardServer] = None
        dashboard_cfg = self.cfg.get('dashboard', {})
        if dashboard_cfg.get('enabled', False):
            self.dashboard = DashboardServer(
                host=dashboard_cfg.get('host', '127.0.0.1'),
                port=dashboard_cfg.get('port', 8080)
            )

    def _init_alert_manager(self) -> None:
        """初始化告警管理器。"""
        alert_log = os.path.join(self.cfg.get('paths', {}).get('log_dir', 'logs'), 'alerts.jsonl')
        self.alert_manager = AlertManager(alert_log_path=alert_log)

    def _init_counters(self) -> None:
        """初始化计数器与锁。"""
        self._recent_errors: deque = deque(maxlen=100)
        self._cycle_count = 0
        self._running = False
        self._last_review_time = 0.0
        interval_hours = self.cfg.get('review', {}).get('interval_hours', 6)
        if not isinstance(interval_hours, (int, float)) or interval_hours < 0:
            interval_hours = 6
        self._review_interval = interval_hours * 3600
        self._lock = threading.RLock()

    def _setup_logging(self) -> None:
        """配置日志系统。"""
        log_cfg = self.cfg.get('logging', {})
        level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
        log_dir = self.cfg.get('paths', {}).get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        handler = self._create_log_handler(log_cfg, log_dir)
        self._log_handler = handler
        self._configure_root_logger(level, handler)

    def _create_log_handler(self, log_cfg: Dict[str, Any], log_dir: str) -> SafeRotatingFileHandler:
        log_file = os.path.join(log_dir, log_cfg.get('file', 'agent.log').split('/')[-1])
        handler = SafeRotatingFileHandler(log_file, maxBytes=log_cfg.get('max_bytes', 10485760), backupCount=log_cfg.get('backup_count', 5))
        handler.setFormatter(self._create_formatter(log_cfg))
        return handler

    def _create_formatter(self, log_cfg: Dict[str, Any]) -> logging.Formatter:
        fmt = log_cfg.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.Formatter(fmt)

    def _configure_root_logger(self, level: int, handler: SafeRotatingFileHandler) -> None:
        root = logging.getLogger()
        root.setLevel(level)
        self._close_existing_file_handlers(root)
        root.addHandler(handler)

    def _close_existing_file_handlers(self, root: logging.Logger) -> None:
        for h in list(root.handlers):
            if isinstance(h, (logging.handlers.RotatingFileHandler, SafeRotatingFileHandler)):
                root.removeHandler(h)
                h.close()

    def _remove_log_handler(self) -> None:
        handler = getattr(self, '_log_handler', None)
        if handler is not None:
            root = logging.getLogger()
            if handler in root.handlers:
                root.removeHandler(handler)
            handler.close()
            self._log_handler = None

    def _next_cycle(self) -> int:
        """获取下一个周期编号。"""
        with self._lock:
            self._cycle_count += 1
            return self._cycle_count

    def _check_emergency(self, cycle: int, state: Dict[str, Any],
                         safety: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查是否进入紧急模式。"""
        if safety['emergency_mode']:
            self.alert_manager.trigger('critical', 'safety', f"Emergency mode triggered at cycle {cycle}: {safety.get('reason', 'unknown')}")
            return self._emergency_response(state, safety, cycle=cycle)
        return None

    def _run_normal_cycle(self, cycle: int, state: Dict[str, Any],
                          flat_state: Dict[str, float],
                          safety: Dict[str, Any]) -> Dict[str, Any]:
        """执行正常周期步骤。"""
        low_preds, mid_preds, high_preds = self._step_predict(cycle)
        errors, actuals = self._step_evaluate(cycle, flat_state, low_preds)
        reward_result, gap_vector = self._step_reward(cycle, state)
        selected_task = self._step_select_task(cycle, state, gap_vector)
        execution_result = self._step_execute(cycle, selected_task, state, high_preds)
        self._post_execute_steps(cycle, state, low_preds, mid_preds, high_preds, actuals, errors, selected_task, reward_result)
        return self._finalize_cycle(cycle, state, low_preds, mid_preds, high_preds, errors, reward_result, selected_task, execution_result, safety)

    def _finalize_cycle(self, cycle: int, state: Dict[str, Any], low_preds: Dict[str, float], mid_preds: Dict[str, float], high_preds: Dict[str, Any], errors: Dict[str, float], reward_result: Dict[str, Any], selected_task: Any, execution_result: Dict[str, Any], safety: Dict[str, Any]) -> Dict[str, Any]:
        review_result = self._step_review(cycle, state)
        snapshot_info = self._step_snapshot(cycle, state, errors, reward_result)
        self._step_cleanup(cycle)
        return self._assemble_cycle_result(cycle, state, low_preds, mid_preds, high_preds, errors, reward_result, selected_task, execution_result, safety, review_result, snapshot_info)

    def _post_execute_steps(self, cycle: int, state: Dict[str, Any],
                            low_preds: Dict[str, float], mid_preds: Dict[str, float], high_preds: Dict[str, Any],
                            actuals: Dict[str, float], errors: Dict[str, float], selected_task: Any, reward_result: Dict[str, Any]) -> None:
        self._step_memory(cycle, state, low_preds, mid_preds, high_preds,
                          actuals, errors, selected_task, reward_result)
        self._step_optimize(cycle, errors)
        self._step_strategy(cycle, state, high_preds, reward_result)

    def _assemble_cycle_result(self, cycle: int, state: Dict[str, Any],
                               low_preds: Dict[str, float],
                               mid_preds: Dict[str, float],
                               high_preds: Dict[str, Any],
                               errors: Dict[str, float],
                               reward_result: Dict[str, Any],
                               selected_task: Any,
                               execution_result: Dict[str, Any],
                               safety: Dict[str, Any],
                               review_result: Dict[str, Any],
                               snapshot_info: Dict[str, Any]) -> Dict[str, Any]:
        """组装周期结果。"""
        return self._build_result_dict(cycle, state, low_preds, mid_preds, high_preds, errors, reward_result, selected_task, execution_result, safety, review_result, snapshot_info)

    def _build_result_dict(self, cycle: int, state: Dict[str, Any],
                           low_preds: Dict[str, float],
                           mid_preds: Dict[str, float],
                           high_preds: Dict[str, Any],
                           errors: Dict[str, float],
                           reward_result: Dict[str, Any],
                           selected_task: Any,
                           execution_result: Dict[str, Any],
                           safety: Dict[str, Any],
                           review_result: Dict[str, Any],
                           snapshot_info: Dict[str, Any]) -> Dict[str, Any]:
        preds = {'low': low_preds, 'mid': mid_preds, 'high': high_preds}
        task = selected_task.to_dict() if selected_task else None
        return dict(cycle=cycle, state=state, predictions=preds, errors=errors, reward=reward_result, task=task, execution=execution_result, safety=safety, review=review_result, snapshot=snapshot_info)

    def _start_cycle(self) -> Tuple[int, float]:
        """启动周期并返回周期编号和开始时间。"""
        cycle = self._next_cycle()
        start = time.time()
        logger.info(f"[Cycle {cycle}] START")
        return cycle, start

    def _run_cycle_body(self, cycle: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """执行周期主体并返回结果数据和安全状态。"""
        state, flat_state = self._step_perceive(cycle)
        safety = self._step_safety(cycle, state)
        emergency = self._check_emergency(cycle, state, safety)
        if emergency is not None:
            return emergency, None, None
        result_data = self._run_normal_cycle(cycle, state, flat_state, safety)
        return result_data, state, safety

    def _finish_cycle(self, cycle: int, result_data: Dict[str, Any],
                      cycle_start: float) -> Dict[str, Any]:
        """完成周期并返回结果。"""
        duration = time.time() - cycle_start
        logger.info(f"[Cycle {cycle}] END, duration={duration:.4f}s")
        result = self._build_result(result_data, duration)
        self._update_dashboard(result)
        return result

    def run_cycle(self) -> Dict[str, Any]:
        """执行一个完整的主循环周期。

        :return: 周期结果字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        current_cycle, cycle_start = self._start_cycle()
        result_data, state, safety = self._run_cycle_body(current_cycle)
        return self._handle_cycle_result(current_cycle, result_data, cycle_start)

    def _handle_cycle_result(self, cycle: int, result_data: Dict[str, Any],
                             cycle_start: float) -> Dict[str, Any]:
        if result_data is None:
            return result_data
        return self._finish_cycle(cycle, result_data, cycle_start)

    def _step_perceive(self, cycle: int) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """感知环境。"""
        logger.info(f"[Cycle {cycle}] STEP 1: perceive")
        state = self.state_vector.perceive()
        flat_state = self.state_vector.flatten(state)
        logger.info(f"[Cycle {cycle}] STEP 1: perceive DONE")
        return state, flat_state

    def _step_safety(self, cycle: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """安全护栏检查。"""
        logger.info(f"[Cycle {cycle}] STEP 2: safety_check")
        safety = self._run_safety_check(cycle)
        if not safety.get('emergency_mode', False):
            self.safety_enforcer.clear_emergency()
        logger.info(f"[Cycle {cycle}] STEP 2: safety_check DONE")
        return safety

    def _run_safety_check(self, cycle: int) -> Dict[str, Any]:
        try:
            return self.safety_enforcer.full_check()
        except Exception as e:
            logger.error(f"[Cycle {cycle}] Safety check failed: {e}")
            return {'emergency_mode': True, 'reason': f'safety_error: {e}'}

    def _step_predict(self, cycle: int) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
        """多层预测。"""
        logger.info(f"[Cycle {cycle}] STEP 3: predict")
        low_preds = self._predict_low(cycle)
        mid_preds = self._predict_mid(cycle)
        high_preds = self._predict_high(cycle, low_preds, mid_preds)
        logger.info(f"[Cycle {cycle}] STEP 3: predict DONE")
        return low_preds, mid_preds, high_preds

    def _predict_low(self, cycle: int) -> Dict[str, float]:
        try:
            return self.low_predictor.predict()
        except Exception as e:
            logger.error(f"[Cycle {cycle}] Low predictor failed: {e}")
            return {'cpu_percent': 0.0, 'mem_percent': 0.0, 'disk_percent': 0.0}

    def _predict_mid(self, cycle: int) -> Dict[str, float]:
        try:
            return self.mid_predictor.predict()
        except Exception as e:
            logger.error(f"[Cycle {cycle}] Mid predictor failed: {e}")
            return {'cpu_percent': 0.0, 'mem_percent': 0.0, 'disk_percent': 0.0}

    def _predict_high(self, cycle: int, low_preds: Dict[str, float],
                      mid_preds: Dict[str, float]) -> Dict[str, Any]:
        try:
            return self.high_predictor.predict(low_preds, mid_preds)
        except Exception as e:
            logger.error(f"[Cycle {cycle}] High predictor failed: {e}")
            return {'recommended_action': 'continue_exploration'}

    def _step_evaluate(self, cycle: int, flat_state: Dict[str, Any], low_preds: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """预测误差评估与修正。"""
        logger.info(f"[Cycle {cycle}] STEP 4: evaluate")
        actuals = self._build_actuals(flat_state)
        errors = self._evaluate_and_update(cycle, low_preds, actuals)
        return errors, actuals

    def _evaluate_and_update(self, cycle: int, low_preds: Dict[str, Any], actuals: Dict[str, float]) -> Dict[str, float]:
        error_result = self.error_evaluator.evaluate(low_preds, actuals)
        self._process_errors(error_result)
        self.low_predictor.update(actuals)
        self.mid_predictor.update(actuals)
        logger.info(f"[Cycle {cycle}] STEP 4: evaluate DONE, mae={error_result['mae']:.4f}")
        return error_result['errors']

    def _build_actuals(self, flat_state: Dict[str, Any]) -> Dict[str, float]:
        return {'cpu_percent': flat_state.get('system.cpu_percent', 0.0), 'mem_percent': flat_state.get('system.mem_percent', 0.0), 'disk_percent': flat_state.get('system.disk_percent', 0.0)}

    def _process_errors(self, error_result: Dict[str, Any]) -> None:
        for err in error_result['errors'].values():
            self._recent_errors.append(err)
        self.state_vector.update_mae(error_result['mae'])

    def _step_reward(self, cycle: int, state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """驱力计算。"""
        logger.info(f"[Cycle {cycle}] STEP 5: reward")
        reward_result = self._compute_reward(cycle, state)
        gap_vector = self._build_gap_vector(reward_result)
        logger.info(f"[Cycle {cycle}] STEP 5: reward DONE, total={reward_result.get('total_reward', 0.0):.4f}")
        return reward_result, gap_vector

    def _compute_reward(self, cycle: int, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            recent_errors_list = list(self._recent_errors)
            return self.reward_engine.compute_total_reward(state, recent_errors_list)
        except Exception as e:
            logger.error(f"[Cycle {cycle}] Reward engine failed: {e}")
            return self._default_reward_result()

    def _default_reward_result(self) -> Dict[str, Any]:
        return {'total_reward': 0.0, 'safe_reward': 0.0, 'gain_reward': 0.0, 'eff_reward': 0.0, 'safe_gap': 0.0, 'gain_gap': 0.0, 'eff_gap': 0.0}

    def _build_gap_vector(self, reward_result: Dict[str, Any]) -> Dict[str, float]:
        return {
            'safe_gap': reward_result.get('safe_gap', 0.0),
            'gain_gap': reward_result.get('gain_gap', 0.0),
            'eff_gap': reward_result.get('eff_gap', 0.0),
        }

    def _step_select_task(self, cycle: int, state: Dict[str, Any], gap_vector: Dict[str, Any]) -> Any:
        """任务池更新与选择。"""
        logger.debug(f"[Cycle {cycle}] STEP 6: task_select")
        self._update_task_pool(gap_vector)
        self._score_tasks(state, gap_vector)
        selected_task = self.task_pool.select_task()
        logger.debug(f"[Cycle {cycle}] STEP 6: task_select DONE, task={selected_task.id if selected_task else None}")
        return selected_task

    def _update_task_pool(self, gap_vector: Dict[str, Any]) -> None:
        new_tasks = self.task_generator.generate(gap_vector)
        self.task_pool.update(new_tasks)

    def _score_tasks(self, state: Dict[str, Any], gap_vector: Dict[str, Any]) -> None:
        scores = self.task_pool.score_all(state, gap_vector)
        for task_id, score in scores.items():
            if task_id in self.task_pool.tasks:
                self.task_pool.tasks[task_id].score = score

    def _step_execute(self, cycle: int, selected_task: Any, state: Dict[str, Any], high_preds: Dict[str, Any]) -> Dict[str, Any]:
        """任务执行。"""
        logger.debug(f"[Cycle {cycle}] STEP 7: execute")
        execution_result = self._safe_execute_task(cycle, selected_task, state, high_preds)
        logger.debug(f"[Cycle {cycle}] STEP 7: execute DONE")
        return execution_result

    def _safe_execute_task(self, cycle: int, selected_task: Any, state: Dict[str, Any], high_preds: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._execute_task(selected_task, state, high_preds)
        except Exception as e:
            logger.error(f"[Cycle {cycle}] Task execution failed: {e}")
            return {'status': 'error', 'task_id': getattr(selected_task, 'id', None), 'success': False, 'reward': 0.0}

    def _step_memory(self, cycle: int, state: Dict[str, Any], low_preds: Dict[str, float], mid_preds: Dict[str, float], high_preds: Dict[str, Any], actuals: Dict[str, float], errors: Dict[str, float], selected_task: Any, reward_result: Dict[str, Any]) -> None:
        """记录记忆。"""
        logger.debug(f"[Cycle {cycle}] STEP 8: memory_store")
        sample = self._build_memory_sample(state, low_preds, mid_preds, high_preds, actuals, errors, selected_task, reward_result)
        self._store_memory_sample(cycle, sample)
        logger.debug(f"[Cycle {cycle}] STEP 8: memory_store DONE")

    def _build_memory_sample(self, state: Dict[str, Any], low_preds: Dict[str, float], mid_preds: Dict[str, float], high_preds: Dict[str, Any], actuals: Dict[str, float], errors: Dict[str, float], selected_task: Any, reward_result: Dict[str, Any]) -> Dict[str, Any]:
        preds = {'low': low_preds, 'mid': mid_preds, 'high': high_preds}
        reward = reward_result.get('total_reward', 0.0)
        return {'timestamp': time.time(), 'state': state, 'predictions': preds, 'actuals': actuals, 'errors': errors, 'task_id': selected_task.id if selected_task else None, 'reward': reward, 'value_score': reward}

    def _store_memory_sample(self, cycle: int, sample: Dict[str, Any]) -> None:
        try:
            self.memory.store_sample(sample)
            if hasattr(self.memory, 'flush'):
                self.memory.flush()
        except Exception as e:
            logger.error(f"[Cycle {cycle}] Memory store failed: {e}")

    def _step_optimize(self, cycle: int, errors: Dict[str, Any]) -> None:
        """参数微修正。"""
        logger.debug(f"[Cycle {cycle}] STEP 9: optimize")
        self.optimizer.optimize(self.low_predictor, errors)
        logger.debug(f"[Cycle {cycle}] STEP 9: optimize DONE")

    def _step_strategy(self, cycle: int, state: Dict[str, Any], high_preds: Dict[str, Any], reward_result: Dict[str, Any]) -> None:
        """高层策略更新。"""
        logger.debug(f"[Cycle {cycle}] STEP 10: strategy_update")
        action = high_preds.get('recommended_action', 'continue_exploration')
        self.high_predictor.update(state, action, reward_result['total_reward'])
        logger.debug(f"[Cycle {cycle}] STEP 10: strategy_update DONE")

    def _step_review(self, cycle: int, state: Dict[str, Any]) -> Optional[Any]:
        """离线复盘触发。"""
        logger.debug(f"[Cycle {cycle}] STEP 11: review_check")
        review_result = self._run_review_if_needed(cycle, state)
        logger.debug(f"[Cycle {cycle}] STEP 11: review_check DONE")
        return review_result

    def _run_review_if_needed(self, cycle: int, state: Dict[str, Any]) -> Optional[Any]:
        if self._should_review(state):
            result = self.review_engine.run_review()
            self._last_review_time = time.time()
            logger.info(f"[Cycle {cycle}] Review completed: {result}")
            return result
        return None

    def _step_snapshot(self, cycle: int, state: Dict[str, Any], errors: Dict[str, Any], reward_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """模型快照与回滚检查。"""
        logger.debug(f"[Cycle {cycle}] STEP 12: snapshot")
        snapshot_info = None
        if cycle % 100 == 0:
            snapshot_info = self._create_snapshot_and_check(cycle, state, errors, reward_result)
        logger.debug(f"[Cycle {cycle}] STEP 12: snapshot DONE")
        return snapshot_info

    def _create_snapshot_and_check(self, cycle: int, state: Dict[str, Any], errors: Dict[str, Any], reward_result: Dict[str, Any]) -> Dict[str, Any]:
        mae = float(sum(errors.values()) / max(len(errors), 1))
        metrics = {'mae': mae, 'coverage': state.get('self', {}).get('coverage_ratio', 0.5), 'eff_score': reward_result['eff_reward']}
        snapshot_path = self.snapshot_manager.create_snapshot(model_state={'cycle': cycle, 'errors': list(errors.values())}, metrics=metrics)
        rollback_result = self.auto_rollback.check_and_rollback(current_mae=mae, restore_callback=self._on_rollback)
        if rollback_result.get('rollback_triggered', False):
            self.alert_manager.trigger('critical', 'rollback', f"Auto-rollback triggered at cycle {cycle}, mae={mae:.4f}")
        return {'path': snapshot_path, 'rollback': rollback_result}

    def _step_cleanup(self, cycle: int) -> None:
        """自动清理记忆。"""
        if cycle % 10 == 0:
            try:
                self.memory.cleanup_old_samples(retention_ratio=0.8)
            except Exception as e:
                logger.warning("AgentLoop._step_cleanup: cleanup_old_samples failed at cycle %s: %s", cycle, e)

    def _build_result(self, result_data: Dict[str, Any],
                      cycle_duration: float) -> Dict[str, Any]:
        """构建周期结果字典。"""
        result_data['duration_seconds'] = round(cycle_duration, 4)
        return result_data

    def _update_dashboard(self, result: Dict[str, Any]) -> None:
        """更新仪表盘。"""
        if self.dashboard is not None:
            self.dashboard.update(result)
            self.dashboard.update_stats(self.get_stats())

    def _execute_task(self, task: Any, state: Dict[str, Any],
                      high_preds: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务并返回结果。

        :param task: 任务对象。
        :type task: Any
        :param state: 当前状态。
        :type state: Dict[str, Any]
        :param high_preds: 高层预测。
        :type high_preds: Dict[str, Any]
        :return: 执行结果。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        if task is None:
            return {'status': 'no_task', 'success': True}
        if self._is_forbidden(task):
            return {'status': 'forbidden', 'success': False}
        return self._execute_valid_task(task, state, high_preds)

    def _execute_valid_task(self, task: Any, state: Dict[str, Any], high_preds: Dict[str, Any]) -> Dict[str, Any]:
        success, reward = self._run_task_action(task, state, high_preds)
        self._update_task_after_execution(task, success)
        self.task_pool.mark_executed(task.id, success, reward)
        self.state_vector.update_task_success(success)
        return {'status': 'executed', 'task_id': task.id, 'success': success, 'reward': reward}

    def _update_task_after_execution(self, task: Any, success: bool) -> None:
        if not self.behavior_guardian.check_exploration(task.id, success):
            logger.warning(f"Task {task.id} frozen due to consecutive failures")
            self.task_pool.freeze_task(task.id)

    def _is_forbidden(self, task: Any) -> bool:
        """检查任务是否被禁止。"""
        if hasattr(task, 'target_path') and task.target_path:
            if not self.behavior_guardian.check_path(task.target_path):
                logger.warning(f"Task {task.id} forbidden path: {task.target_path}")
                self.task_pool.freeze_task(task.id)
                return True
        return False

    def _run_task_action(self, task: Any, state: Dict[str, Any], high_preds: Dict[str, Any]) -> Tuple[bool, float]:
        """运行任务动作并返回结果。"""
        try:
            success = self._execute_action(task, state, high_preds)
            return success, 1.0 if success else 0.0
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return False, 0.0

    def _execute_action(self, task: Any, state: Dict[str, Any], high_preds: Dict[str, Any]) -> bool:
        action = high_preds.get('recommended_action', 'continue_exploration')
        if action == 'emergency_safeguard':
            return self._do_safeguard(task, state)
        elif action == 'increase_monitoring':
            return self._do_monitoring(task, state)
        return self._do_exploration(task, state)

    def _do_safeguard(self, task: Any, state: Dict[str, Any]) -> bool:
        """执行保护动作。"""
        logger.info(f"Safeguard action for task {task.id}")
        return True

    def _do_monitoring(self, task: Any, state: Dict[str, Any]) -> bool:
        """执行监控动作。"""
        logger.info(f"Monitoring action for task {task.id}")
        return True

    def _do_exploration(self, task: Any, state: Dict[str, Any]) -> bool:
        """执行探索动作。"""
        logger.info(f"Exploration action for task {task.id}: {task.name}")
        return True

    def _emergency_response(self, state: Dict[str, Any],
                            safety: Dict[str, Any],
                            cycle: int = 0) -> Dict[str, Any]:
        """紧急模式响应。

        :param state: 当前状态。
        :type state: Dict[str, Any]
        :param safety: 安全检查结果。
        :type safety: Dict[str, Any]
        :param cycle: 周期数。
        :type cycle: int
        :return: 紧急响应结果。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        logger.warning(f"[Cycle {cycle}] EMERGENCY RESPONSE triggered")
        return self._build_emergency_result(cycle, state, safety)

    def _build_emergency_result(self, cycle: int, state: Dict[str, Any], safety: Dict[str, Any]) -> Dict[str, Any]:
        return {'cycle': cycle, 'state': state, 'predictions': None, 'errors': {}, 'reward': {'total_reward': 0.0}, 'task': None, 'execution': {'status': 'emergency', 'success': False}, 'safety': safety, 'review': None, 'snapshot': None, 'duration_seconds': 0.0}

    def _should_review(self, state: Dict[str, Any]) -> bool:
        """判断是否应该复盘。

        :param state: 当前状态。
        :type state: Dict[str, Any]
        :return: 是否应该复盘。
        :rtype: bool
        :raises: 无。
        """
        if not self._review_interval_passed():
            return False
        return self._check_review_conditions(state)

    def _review_interval_passed(self) -> bool:
        return time.time() - self._last_review_time >= self._review_interval

    def _check_review_conditions(self, state: Dict[str, Any]) -> bool:
        sample_count = self.memory.get_sample_count()
        idle_cpu = self.cfg.get('review', {}).get('idle_cpu_threshold', 10.0)
        system_cpu = state.get('system', {}).get('cpu_percent', 100.0)
        return self.review_engine.should_review(sample_count, 100.0 - system_cpu)

    def _on_rollback(self, snapshot_path: str) -> None:
        """回滚回调。"""
        logger.warning(f"Auto-rollback triggered to snapshot: {snapshot_path}")

    def run(self, max_cycles: Optional[int] = None,
            sleep_interval: float = 1.0) -> None:
        """持续运行主循环。

        :param max_cycles: 最大周期数。
        :type max_cycles: Optional[int]
        :param sleep_interval: 周期间隔（秒）。
        :type sleep_interval: float
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        self._prepare_run()
        try:
            self._run_cycles(max_cycles, sleep_interval)
        except KeyboardInterrupt:
            logger.info("Agent stopped by user")
        finally:
            self.stop()

    def _prepare_run(self) -> None:
        self._running = True
        self._start_dashboard()
        logger.info("Agent main loop started")

    def _start_dashboard(self) -> None:
        if self.dashboard is not None:
            self.dashboard.set_alert_manager(self.alert_manager)
            self.dashboard.start()

    def _run_cycles(self, max_cycles: Optional[int], sleep_interval: float) -> None:
        while self._running:
            try:
                result = self.run_cycle()
                logger.debug(f"Cycle {result['cycle']} completed in {result['duration_seconds']}s")
            except Exception as e:
                # 修复阶段116发现的bug：单步异常不应崩溃整个主循环
                logger.error(f"Cycle {self._cycle_count} failed with exception: {type(e).__name__}: {e}", exc_info=True)
                self._cycle_count += 1
            if max_cycles is not None and self._cycle_count >= max_cycles:
                break
            time.sleep(sleep_interval)

    def stop(self) -> None:
        """停止主循环。

        :return: 无。
        :rtype: None
        :raises: 无。
        """
        self._running = False
        self._cleanup_resources()
        logger.info("Agent main loop stopped")

    def _cleanup_resources(self) -> None:
        # 修复Bug2: 使用try/finally确保异常路径下资源也被释放
        try:
            if hasattr(self.memory, 'flush'):
                self.memory.flush()
        except Exception as e:
            logger.error("AgentLoop._cleanup_resources: flush failed: %s", e)
        finally:
            try:
                self.memory.close()
            except Exception as e:
                logger.error("AgentLoop._cleanup_resources: close failed: %s", e)
            try:
                if self.dashboard is not None:
                    self.dashboard.stop()
            except Exception as e:
                logger.error("AgentLoop._cleanup_resources: dashboard stop failed: %s", e)
            try:
                self._remove_log_handler()
            except Exception as e:
                logger.error("AgentLoop._cleanup_resources: remove log handler failed: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        :return: 统计字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        return self._build_stats_dict()

    def _build_stats_dict(self) -> Dict[str, Any]:
        return {
            'cycle_count': self._cycle_count,
            'task_pool': self.task_pool.get_stats(),
            'memory_samples': self.memory.get_sample_count(),
            'recent_mae': self.state_vector._get_recent_mae(),
            'recent_success_rate': self.state_vector._get_recent_success_rate(),
        }
