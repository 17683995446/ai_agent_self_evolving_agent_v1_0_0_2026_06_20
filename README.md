# Self-Evolving Agent

基于第一性原理的**自驱永续进化预测修正 Agent 系统**。系统仿生大脑模型：感知环境 → 内生动机 → 自主确定行为 → 多层预测 → 比对现实误差 → 修正模型 → 迭代进化。

## 核心特性

- **自驱运行**：零人工输入，自主孵化任务并执行
- **多层预测**：底层高频微预测、中层趋势预测、高层抽象策略
- **双闭环进化**：在线实时预测修正 + 离线复盘模型进化
- **安全护栏**：资源约束（CPU/内存上限）、行为安全约束、探索止损
- **多 Agent 协作**：自主学会操作多个 agent 完成任务
- **技能自增强**：自主学会增强自己的技能
- **长任务自驱动**：自主学会自己驱动自己完成长任务

## 快速开始

### 环境要求

- Python 3.10+
- Linux / macOS / Windows

### 安装

```bash
git clone <repo-url> self_evolving_agent
cd self_evolving_agent

# 方式一：可编辑安装（推荐，注册 self-evolving-agent 命令）
pip install -e .

# 方式二：仅安装依赖
pip install -r requirements.txt
```

安装后可直接使用 `self-evolving-agent` 命令，或通过 `python -m src.cli.main` 调用。

### 30 秒体验

```bash
# 一键快速体验（自动执行：健康检查 → 初始化配置 → 单次循环 → 查看状态）
self-evolving-agent quickstart
```

或分步执行：

```bash
# 1. 初始化默认配置
self-evolving-agent config init

# 2. 执行单次循环（观察输出）
self-evolving-agent cycle

# 3. 启动持续运行（Ctrl+C 停止）
self-evolving-agent start --cycles 10
```

### 查看状态

```bash
# 文本格式
self-evolving-agent status

# JSON 格式（便于脚本处理）
self-evolving-agent status --json
```

## CLI 命令一览

```bash
self-evolving-agent <command> [options]
```

| 命令 | 说明 | 示例 |
|------|------|------|
| `start` | 启动 Agent 主循环 | `start --cycles 10 --interval 1.0` |
| `stop` | 停止运行中的 Agent | `stop` |
| `status` | 查看 Agent 状态 | `status --json` |
| `cycle` | 执行单次循环（调试用） | `cycle --json` |
| `doctor` | 健康检查（依赖/配置/路径） | `doctor` |
| `logs` | 查看最近日志 | `logs --lines 20` |
| `quickstart` | 一键体验（doctor+init+cycle+status） | `quickstart` |
| `dashboard` | 启动监控仪表盘 Web UI | `dashboard --port 8080` |
| `cleanup` | 清理旧快照/日志/数据库碎片 | `cleanup --dry-run` |
| `config show` | 查看当前配置 | `config show` |
| `config validate` | 校验配置合法性 | `config validate` |
| `config init` | 生成默认配置文件 | `config init --force` |
| `config wizard` | 交互式配置向导 | `config wizard` |

**全局选项**：

- `--version, -V`：查看版本号
- `--config, -c`：指定配置文件路径（默认按 `AGENT_ENV` 环境变量选择 dev/test/prod）
- `--json`：以 JSON 格式输出（适用于 `status`、`cycle`）
- `--verbose, -v`：详细输出模式（显示周期分解详情，适用于 `cycle`）
- `--no-color`：禁用彩色输出（TTY 环境下自动启用颜色，遵循 `NO_COLOR` 环境变量规范）

**环境变量**：

- `AGENT_ENV`：选择配置环境，`dev` / `test` / `prod`（默认 `prod`）
- `NO_COLOR`：设置此变量时禁用彩色输出（遵循 [no-color.org](https://no-color.org/) 规范）

### 监控仪表盘

启动 Web 仪表盘，通过浏览器实时查看 Agent 运行状态（MAE、成功率、记忆样本、历史趋势图）：

```bash
self-evolving-agent dashboard --port 8080
# 浏览器访问 http://127.0.0.1:8080
```

仪表盘读取 `metrics.json` 和快照索引，无需 Agent 在线即可查看历史数据。按 `Ctrl+C` 停止。

## 配置说明

配置文件为 YAML 格式，主要分段：

| 分段 | 作用 | 关键字段 |
|------|------|----------|
| `goals` | 进化目标开关 | `precision_evolution` / `coverage_evolution` / `efficiency_evolution` |
| `base_weights` | 驱力权重 | `safe` / `gain` / `eff`（和为 1.0） |
| `resource_limits` | 资源上限 | `max_cpu_percent` / `max_mem_percent` |
| `safety` | 安全约束 | `forbidden_paths` / `max_consecutive_failures` |
| `review` | 离线复盘 | `interval_hours` / `min_samples` |
| `memory` | 记忆管理 | `max_short_term_samples` |
| `logging` | 日志 | `level` / `file` / `max_bytes` |
| `paths` | 数据路径 | `data_dir` / `snapshot_dir` / `log_dir` |

三套预置配置位于 `config/` 目录：`config.dev.yaml`（开发）、`config.test.yaml`（测试）、`config.prod.yaml`（生产）。

## 项目结构

```
self_evolving_agent/
├── src/
│   ├── core/               # 核心引擎（主循环、配置、状态机、事件总线）
│   ├── perception/         # 感知层（系统/日志/文件传感器）
│   ├── prediction/         # 预测层（ARIMA、指数平滑、分层预测）
│   ├── drive/              # 驱力层（安全/收益/能效驱力）
│   ├── task/               # 任务层（生成、评分、选择、执行）
│   ├── memory/             # 记忆层（短期/长期记忆、复盘引擎）
│   ├── safety/             # 安全层（资源护栏、行为护栏）
│   ├── optimization/       # 优化层（快照、自动回滚、微优化）
│   ├── planning/           # 规划层（长任务规划、多Agent编排）
│   ├── multi_agent/        # 多Agent协作（编排器、工作节点）
│   ├── skills/             # 技能层（技能注册、学习、组合、评估）
│   ├── dashboard/          # 监控仪表盘
│   ├── monitoring/         # 监控告警
│   └── cli/                # 命令行入口
├── config/                 # 配置文件（dev/test/prod）
├── tests/                  # 测试套件
├── docs/                   # 文档
├── examples/               # 示例代码
├── scripts/                # 启停脚本
├── Dockerfile              # 容器构建
└── docker-compose.yml      # 容器编排
```

## 运行测试

```bash
# 全量测试
python -m pytest tests/ -q

# 指定模块测试
python -m pytest tests/test_agent_loop.py -q

# 查看覆盖率
python -m pytest tests/ --cov=src -q
```

## Docker 部署

```bash
docker build -t self-evolving-agent .
docker run -d --name agent self-evolving-agent
```

或使用 docker-compose：

```bash
docker-compose up -d
```

## 三大应用场景

### 1. 自主学会操作多个 agent 完成任务

系统通过 `MultiAgentOrchestrator` 编排多个具备不同能力的 `AgentWorker`，根据任务需求自动匹配最合适的 worker，实现协作完成任务。

```bash
python examples/multi_agent_long_task_demo.py
```

### 2. 自主学会增强自己的技能

`SkillRegistry` 管理技能库，`SkillLearner` 从执行经验中学习新技能，`CapabilityAssessor` 评估能力差距并驱动技能增强。

### 3. 自主学会自己驱动自己完成长任务

`LongTaskPlanner` 将长任务分解为可检查点的子任务，`MultiAgentLongTaskRunner` 驱动执行，`LongTaskReviewer` 复盘改进，支持中断恢复。

## 入门示例

`examples/` 目录提供渐进式入门示例：

| 示例 | 说明 | 运行命令 |
|------|------|----------|
| `01_single_cycle.py` | 最简单的单周期运行 | `python examples/01_single_cycle.py` |
| `02_multi_cycle.py` | 多周期连续运行与状态监控 | `python examples/02_multi_cycle.py` |
| `03_custom_config.py` | 自定义配置运行 | `python examples/03_custom_config.py` |
| `multi_agent_long_task_demo.py` | 多Agent长任务端到端演示 | `python examples/multi_agent_long_task_demo.py` |

## 文档

- [项目白皮书](docs/PROJECT_WHITEPAPER.md)：系统架构与设计原理
- [交付清单](docs/DELIVERY_CHECKLIST.md)：验收标准
- [API 文档](docs/api/openapi.yaml)：仪表盘 API 规范

## 技术栈

- **预测模型**：指数平滑、ARIMA（statsmodels）
- **状态感知**：psutil（系统指标）、watchdog（文件监控）
- **数据存储**：SQLite（短期记忆）、JSON（快照）
- **监控**：HTTP 仪表盘、日志聚合、告警管理
