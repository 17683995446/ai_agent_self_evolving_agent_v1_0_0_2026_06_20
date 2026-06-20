import abc
import os
import logging
import threading
import yaml
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class _ConfigMeta(abc.ABCMeta):
    """配置类元类：拦截 ``Config._instance`` 与 ``Config._config`` 类属性赋值，
    向后兼容旧测试代码的重置操作。

    - ``Config._instance = None`` 等价于清空当前类的单例缓存
    - ``Config._config = {...`` 等价于设置当前单例实例的配置字典
    """

    def __setattr__(cls, name, value):
        if name == '_instance' and value is None:
            cls._instances.pop(cls, None)
            return
        if name == '_config':
            instance = cls._instances.get(cls)
            if instance is not None:
                instance._config = value
            return
        super().__setattr__(name, value)

    def __getattr__(cls, name):
        """对未定义的类属性 ``_instance`` 提供向后兼容读取。"""
        if name == '_instance':
            return cls._instances.get(cls)
        if name == '_config':
            instance = cls._instances.get(cls)
            return instance._config if instance is not None else {}
        raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")


class BaseConfig(metaclass=_ConfigMeta):
    """配置管理抽象基类。

    提供配置加载、获取、设置和导出的抽象接口，支持单例模式。
    每个子类拥有独立的单例实例与配置字典，避免跨子类共享污染。
    """

    _instances: Dict[type, 'BaseConfig'] = {}
    _lock = threading.RLock()
    _singleton = True

    def __new__(cls, config_path: Optional[str] = None):
        """创建或返回单例实例。

        :param config_path: 配置文件路径。
        :type config_path: Optional[str]
        :return: 单例实例。
        :rtype: BaseConfig
        """
        if not cls._singleton:
            instance = super().__new__(cls)
            instance._config = {}
            return instance
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                instance._config = {}
                cls._instances[cls] = instance
            return cls._instances[cls]

    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器。

        :param config_path: 配置文件路径，提供则自动加载。
        :type config_path: Optional[str]
        """
        if not hasattr(self, '_config'):
            self._config = {}
        if config_path:
            self.load(config_path)

    @abc.abstractmethod
    def load(self, config_path: str) -> None:
        """加载配置文件。

        :param config_path: 配置文件路径。
        :type config_path: str
        :return: 无。
        :rtype: None
        :raises FileNotFoundError: 文件不存在时抛出。
        """

    @abc.abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项。

        :param key: 点分键名。
        :type key: str
        :param default: 默认值。
        :type default: Any
        :return: 配置值或默认值。
        :rtype: Any
        :raises: 无。
        """

    @abc.abstractmethod
    def set(self, key: str, value: Any) -> None:
        """设置配置项。

        :param key: 点分键名。
        :type key: str
        :param value: 要设置的值。
        :type value: Any
        :return: 无。
        :rtype: None
        :raises: 无。
        """

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """导出配置字典。

        :return: 配置字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """


class Config(BaseConfig):
    """配置管理类（单例模式）。

    基于 YAML 配置文件，支持点分键名访问、线程安全的读写操作。
    """

    def load(self, config_path: str) -> None:
        """加载配置文件。

        :param config_path: 配置文件路径。
        :type config_path: str
        :return: 无。
        :rtype: None
        :raises FileNotFoundError: 文件不存在时抛出。
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with self._lock:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            self._loaded_path = config_path

    def reload(self) -> None:
        """重新加载配置文件（热重载）。

        重新读取上次 load 的配置文件。如果文件不存在或解析失败，
        保留旧配置不崩溃。

        :return: 无。
        :rtype: None
        :raises FileNotFoundError: 文件不存在时抛出，但旧配置保留。
        """
        config_path = getattr(self, '_loaded_path', None)
        if config_path is None:
            logger.warning("Config.reload: no config path recorded, nothing to reload")
            return
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error("Config.reload: failed to parse %s: %s", config_path, e)
            return
        with self._lock:
            self._config = new_config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项。

        :param key: 点分键名。
        :type key: str
        :param default: 默认值。
        :type default: Any
        :return: 配置值或默认值。
        :rtype: Any
        :raises: 无。
        """
        with self._lock:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

    def set(self, key: str, value: Any) -> None:
        """设置配置项。

        :param key: 点分键名。
        :type key: str
        :param value: 要设置的值。
        :type value: Any
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        with self._lock:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                if k not in config or not isinstance(config[k], dict):
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value

    def get_base_weights(self) -> tuple:
        """获取基础权重。

        :param: 无
        :type: 无
        :return: (safe, gain, eff) 权重元组。
        :rtype: tuple
        :raises: 无。
        """
        with self._lock:
            weights = self._config.get('base_weights', {})
            return (
                weights.get('safe', 0.4),
                weights.get('gain', 0.4),
                weights.get('eff', 0.2)
            )

    def get_resource_limits(self) -> Dict[str, float]:
        """获取资源限制配置。

        :param: 无
        :type: 无
        :return: 资源限制字典。
        :rtype: Dict[str, float]
        :raises: 无。
        """
        with self._lock:
            return dict(self._config.get('resource_limits', {
                'max_cpu_percent': 20.0,
                'max_mem_percent': 20.0,
                'max_db_size_mb': 500.0,
                'max_snapshot_count': 10.0
            }))

    def get_exploration_config(self) -> Dict[str, Any]:
        """获取探索配置。

        :param: 无
        :type: 无
        :return: 探索参数字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        with self._lock:
            return dict(self._config.get('exploration', {
                'base_ratio': 0.3,
                'stable_threshold': 0.05,
                'emergency_ratio': 0.1,
                'novel_ratio': 0.4
            }))

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置。

        :param: 无
        :type: 无
        :return: 模型参数字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        with self._lock:
            return dict(self._config.get('models', {
                'smoothing_alpha': 0.3,
                'smoothing_alpha_min': 0.05,
                'smoothing_alpha_max': 0.8,
                'arima_order': [1, 1, 1],
                'prediction_horizon': 1
            }))

    def get_safety_config(self) -> Dict[str, Any]:
        """获取安全配置。

        :param: 无
        :type: 无
        :return: 安全参数字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        with self._lock:
            return dict(self._config.get('safety', {
                'max_consecutive_failures': 5,
                'forbidden_paths': [],
                'max_model_restructure_iterations': 3
            }))

    def get_memory_config(self) -> Dict[str, Any]:
        """获取内存配置。

        :param: 无
        :type: 无
        :return: 内存参数字典。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        with self._lock:
            return dict(self._config.get('memory', {
                'max_short_term_samples': 10000,
                'value_retention_ratio': 0.3,
                'history_window': 100
            }))

    def get_paths(self) -> Dict[str, str]:
        """获取路径配置。

        :param: 无
        :type: 无
        :return: 路径字典。
        :rtype: Dict[str, str]
        :raises: 无。
        """
        with self._lock:
            return dict(self._config.get('paths', {
                'data_dir': 'data',
                'snapshot_dir': 'snapshots',
                'log_dir': 'logs'
            }))

    def ensure_paths(self) -> None:
        """确保配置中的路径已创建。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        :raises: 无。
        """
        paths = self.get_paths()
        for key, path in paths.items():
            os.makedirs(path, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """导出完整配置字典。

        :return: 配置字典副本。
        :rtype: Dict[str, Any]
        :raises: 无。
        """
        with self._lock:
            return dict(self._config)
