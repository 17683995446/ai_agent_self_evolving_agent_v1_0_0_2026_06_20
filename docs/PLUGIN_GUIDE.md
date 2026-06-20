# 插件接入指南

本指南介绍如何扩展 Self-Evolving Agent 系统。系统提供 8 个扩展点，覆盖技能、任务、预测、感知、事件、告警、多 Agent 编排和守护进程。

## 目录

- [1. 技能系统（Skill）](#1-技能系统skill)
- [2. 任务模板（Task Template）](#2-任务模板task-template)
- [3. 预测模型（Predictor）](#3-预测模型predictor)
- [4. 传感器（Sensor）](#4-传感器sensor)
- [5. 事件总线（Event Bus）](#5-事件总线event-bus)
- [6. 告警处理器（Alert Handler）](#6-告警处理器alert-handler)
- [7. 多 Agent Worker](#7-多-agent-worker)
- [8. 守护进程 Hook](#8-守护进程-hook)
- [扩展点速查表](#扩展点速查表)

---

## 1. 技能系统（Skill）

技能是 Agent 执行任务的能力单元。系统支持自动学习新技能和手动注册。

### 1.1 手动注册技能

```python
from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry

registry = SkillRegistry()

skill = Skill(
    name="collect_metrics",
    description="收集系统性能指标",
    capabilities=["monitoring", "metrics"],
    action=lambda task: {"cpu": 45.2, "mem": 60.1},
)

registry.register(skill)
```

### 1.2 自动学习技能（推荐）

通过 `SkillLearner` 从任务执行中自动学习：

```python
from src.skills.skill_learner import SkillLearner

learner = SkillLearner(skill_registry=registry)

# 执行任务后自动注册技能
learner.auto_register(
    task=my_task,
    action=lambda t: perform_action(t),
    description="从任务中学习的技能",
    capabilities=["auto_learned"],
)
```

### 1.3 复合技能

将多个已有技能组合为复合技能：

```python
from src.skills.skill_composer import SkillComposer

composer = SkillComposer(skill_registry=registry)
composer.compose(
    pattern=["collect_metrics", "analyze_data"],
    name="monitor_and_analyze",
    description="收集并分析指标",
)
```

### 1.4 技能持久化

技能可跨会话保存和加载：

```python
from src.skills.skill_persistence import SkillPersistence

persistence = SkillPersistence(data_dir="data")
persistence.save(registry)        # 保存
persistence.load(registry)        # 加载
```

---

## 2. 任务模板（Task Template）

任务类型为固定枚举：`SAFE`（安全）、`GAIN`（收益）、`EFF`（能效）、`EXPLORE`（探索）。可通过模板扩展任务生成器。

### 2.1 添加任务模板

```python
from src.task.task_generator import TaskGenerator

generator = TaskGenerator()

# 添加安全类任务模板
generator.add_safe_template(
    name="check_disk_usage",
    action="check_disk",
)

# 添加收益类任务模板
generator.add_gain_template(
    name="optimize_model",
    action="optimize",
)

# 添加能效类任务模板
generator.add_eff_template(
    name="reduce_memory",
    action="reduce_mem",
)
```

### 2.2 直接创建任务实例

```python
from src.task.task_definition import Task
from src.core.constants import TaskType
from src.task.task_pool import TaskPool

task = Task(
    id="custom_001",
    type=TaskType.GAIN,
    name="custom_task",
    description="自定义任务",
    action="custom_action",
    params={"key": "value"},
)

pool = TaskPool()
pool.update([task])
```

---

## 3. 预测模型（Predictor）

预测模型采用工厂方法字典模式，是系统最完整的扩展点。

### 3.1 创建新预测器

继承 `BasePredictor`，实现 `predict` 和 `update` 方法：

```python
from src.prediction.base_predictor import BasePredictor

class LinearRegressionPredictor(BasePredictor):
    """线性回归预测器示例。"""

    def __init__(self):
        self._data: list[float] = []
        self._slope = 0.0
        self._intercept = 0.0

    def predict(self, steps: int = 1) -> list[float]:
        """预测未来 N 步的值。"""
        if len(self._data) < 2:
            return [self._data[-1] if self._data else 0.0] * steps
        n = len(self._data)
        x_mean = sum(range(n)) / n
        y_mean = sum(self._data) / n
        # 简单线性回归
        ...
        return [self._slope * (n + i) + self._intercept for i in range(steps)]

    def update(self, actual: float) -> None:
        """用新观测值更新模型。"""
        self._data.append(actual)
        if len(self._data) > 100:
            self._data = self._data[-100:]
```

### 3.2 注册到工厂

在 `MultiPredictor` 中注册新模型：

```python
# src/prediction/multi_predictor.py

class MultiPredictor(BaseMultiPredictor):
    def _model_creators(self) -> dict:
        creators = super()._model_creators()
        creators['linear_regression'] = self._create_linear_regression
        return creators

    def _create_linear_regression(self):
        return LinearRegressionPredictor()
```

### 3.3 使用新预测器

```python
predictor = MultiPredictor(model_type='linear_regression')
predictor.add_metric('cpu_percent')
predictor.update(45.2)
predictions = predictor.predict(steps=5)
```

### 3.4 内置预测器

| model_type | 类名 | 说明 |
|------------|------|------|
| `exponential` | `ExponentialSmoother` | 指数平滑 |
| `adaptive_exponential` | `AdaptiveExponentialSmoother` | 自适应指数平滑 |
| `sma` | `SimpleMovingAverage` | 简单移动平均 |
| `arima` | `ARIMAModel` | ARIMA 模型 |

---

## 4. 传感器（Sensor）

传感器负责感知外部环境。系统提供三种内置传感器：系统、文件、日志。

### 4.1 创建新传感器

继承对应基类，实现 `perceive()` 方法：

```python
from src.perception.system_sensor import BaseSystemSensor

class NetworkSensor(BaseSystemSensor):
    """网络流量传感器。"""

    def perceive(self) -> dict:
        """返回网络指标。"""
        import psutil
        net = psutil.net_io_counters()
        return {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv,
        }
```

### 4.2 动态扩展监听目标

文件传感器和日志传感器支持运行时添加监听路径：

```python
from src.perception.file_sensor import FileSensor
from src.perception.log_sensor import LogSensor

file_sensor = FileSensor()
file_sensor.add_watch_path("/var/log/myapp/")
file_sensor.remove_watch_path("/tmp/old_watch/")

log_sensor = LogSensor()
log_sensor.add_log_path("/var/log/myapp/output.log")
```

### 4.3 集成到状态向量

`StateVector` 在构造时硬编码实例化三个传感器。要替换为新传感器，继承 `BaseStateVector`：

```python
from src.perception.state_vector import BaseStateVector

class CustomStateVector(BaseStateVector):
    def perceive(self) -> dict:
        data = super().perceive() if hasattr(super(), 'perceive') else {}
        data.update(self._custom_sensor.perceive())
        return data
```

---

## 5. 事件总线（Event Bus）

事件总线支持发布-订阅模式，基于异步 `asyncio` 实现。订阅器需为 `async` 函数，事件通过 `Event` 对象传递。

### 5.1 订阅事件

```python
import asyncio
from src.core.event_bus import EventBus, Event

bus = EventBus()

# 订阅特定事件：handler 必须是 async 函数，接收 Event 对象
async def on_cycle_completed(event: Event) -> None:
    print(f"Cycle done: {event.payload}")  # 注意：字段是 payload，不是 data

bus.subscribe("cycle_completed", on_cycle_completed)

# 通配符订阅（接收所有事件）
async def log_all_events(event: Event) -> None:
    log_event(event)

bus.subscribe("*", log_all_events)
```

### 5.2 发布事件

```python
import asyncio
from src.core.event_bus import EventBus, Event

bus = EventBus()

async def main() -> None:
    # publish 是 async 方法，接收单个 Event 对象
    event = Event(type="custom_event", payload={"key": "value"}, source="my_plugin")
    await bus.publish(event)

asyncio.run(main())
```

### 5.3 取消订阅

```python
async def handler(event: Event) -> None:
    process(event)

bus.subscribe("cycle_completed", handler)
# ...
bus.unsubscribe("cycle_completed", handler)
```

---

## 6. 告警处理器（Alert Handler）

告警管理器支持自定义处理器，在告警触发时执行自定义逻辑。

### 6.1 添加告警处理器

```python
from src.monitoring.alert_manager import AlertManager

alert_mgr = AlertManager()

def email_handler(alert):
    """告警触发时发送邮件。"""
    send_email(
        to="ops@example.com",
        subject=f"Alert: {alert.severity}",  # 注意：字段是 severity，不是 level
        body=alert.message,
    )

alert_mgr.add_handler(email_handler)
```

### 6.2 内置处理器

- `_log_handler`：将告警写入日志文件
- `_file_handler`：将告警写入独立告警文件（可选）

---

## 7. 多 Agent Worker

多 Agent 编排器支持动态注册/注销 Worker，支持三种调度策略。

### 7.1 注册 Worker

```python
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker

orchestrator = MultiAgentOrchestrator()

worker = AgentWorker(worker_id="worker_01")
worker.add_capability("data_analysis")
worker.add_capability("report_generation")

orchestrator.register_worker(worker)
```

### 7.2 切换调度策略

```python
# 支持的策略：round_robin / capability_match / least_loaded
orchestrator.set_strategy("capability_match")
```

### 7.3 绑定技能注册中心

Worker 可绑定技能注册中心，实现技能共享：

```python
worker.attach_skill_registry(registry)
```

---

## 8. 守护进程 Hook

守护进程管理器支持注册工作函数和关闭钩子。

### 8.1 注册工作函数

```python
from src.core.daemon import DaemonManager

daemon = DaemonManager()

def main_worker():
    while True:
        do_work()
        time.sleep(1)

daemon.register_worker(main_worker)
```

### 8.2 注册关闭钩子

```python
def cleanup():
    print("Cleaning up resources...")
    close_connections()
    flush_buffers()

daemon.register_shutdown_hook(cleanup)
```

### 8.3 便捷启动函数

```python
from src.core.daemon import run_daemon_command

run_daemon_command(
    command="start",
    worker=main_worker,
    pid_file="data/daemon.pid",
)
```

---

## 扩展点速查表

| 扩展点 | 基类 | 注册方法 | 文件位置 |
|--------|------|---------|---------|
| 技能 | `Skill`（普通类） | `SkillRegistry.register()` | `src/skills/` |
| 任务模板 | `BaseTaskGenerator` | `TaskGenerator.add_*_template()` | `src/task/` |
| 预测模型 | `BasePredictor` | `MultiPredictor._model_creators()` | `src/prediction/` |
| 传感器 | `Base*Sensor` | 继承基类 / `add_watch_path()` | `src/perception/` |
| 事件处理 | `BaseEventBus` | `EventBus.subscribe()` | `src/core/event_bus.py` |
| 告警处理 | `_AlertManager` | `AlertManager.add_handler()` | `src/monitoring/alert_manager.py` |
| 多 Agent Worker | `AgentWorker`（普通类） | `MultiAgentOrchestrator.register_worker()` | `src/multi_agent/` |
| 守护进程 Hook | `BaseDaemonManager` | `register_worker()` / `register_shutdown_hook()` | `src/core/daemon.py` |

## 设计原则

1. **抽象基类统一使用 `abc.ABC` + `@abc.abstractmethod`**：所有扩展点均通过继承抽象基类实现，保证接口一致性。
2. **线程安全注册中心**：`SkillRegistry`、`EventBus`、`AlertManager` 等注册中心使用 `threading.RLock` 保证并发安全。
3. **工厂方法字典模式**：`MultiPredictor` 使用工厂字典注册预测器，是添加新模型的标准方式。
4. **向后兼容**：所有新增参数均有默认值，不影响现有调用方。
