# 常见问题（FAQ）

## 配置相关

### Q1: 如何选择运行模式（conservative/balanced/aggressive）？

通过 `config wizard` 交互式选择，或直接修改配置文件：

| 模式 | CPU上限 | 内存上限 | 探索率 | 适用场景 |
|------|---------|---------|--------|----------|
| conservative | 10% | 10% | 0.2 | 共享服务器、生产环境 |
| balanced | 20% | 20% | 0.3 | 日常使用（默认） |
| aggressive | 50% | 50% | 0.5 | 专用机器、实验环境 |

```bash
self-evolving-agent config wizard
```

### Q2: base_weights 三项和必须为 1.0 吗？

是的。`base_weights` 的 `safe`、`gain`、`eff` 三项权重之和必须为 1.0，否则 `config validate` 会报错。

```yaml
base_weights:
  safe: 0.4   # 安全驱力权重
  gain: 0.4   # 收益驱力权重
  eff: 0.2    # 效率驱力权重
  # 三项和 = 0.4 + 0.4 + 0.2 = 1.0 ✓
```

### Q3: 如何切换 JSON 结构化日志？

修改配置文件中的 `logging.format_type`：

```yaml
logging:
  format_type: json  # text 或 json
```

JSON 格式日志每行一个 JSON 对象，包含 `timestamp`、`level`、`logger`、`message`、`module`、`func`、`line` 等字段，便于 ELK/Loki/CloudWatch 等日志聚合系统检索。

### Q4: 如何按模块设置不同日志级别？

在配置文件中添加 `logging.loggers` 映射：

```yaml
logging:
  level: INFO  # 全局默认级别
  loggers:
    src.core: DEBUG        # 核心模块输出 DEBUG
    src.optimization: DEBUG # 优化模块输出 DEBUG
    src.cli: INFO           # CLI 模块保持 INFO
    src.perception: WARNING # 感知模块仅输出 WARNING
```

模块日志器默认 `propagate=True`，日志仍会传播到 root handler 输出到文件。

### Q5: forbidden_paths 默认禁止哪些路径？如何修改？

默认禁止以下系统路径：

```yaml
safety:
  forbidden_paths:
    - /boot
    - /etc
    - /proc
    - /sys
    - /dev
    - /bin
    - /sbin
    - /lib
    - /lib64
    - /usr/bin
    - /usr/sbin
```

如需添加自定义禁止路径：

```yaml
safety:
  forbidden_paths:
    - /etc
    - /proc
    # 添加自定义路径
    - /home/user/secrets
    - /var/lib/mysql
```

> **安全提示**：切勿移除 `/etc`、`/proc`、`/sys` 等系统路径的禁止配置。

## 仪表盘相关

### Q6: dashboard 如何安全地远程访问？

```bash
# 1. 绑定所有网卡 + 设置认证令牌
self-evolving-agent dashboard --host 0.0.0.0 --auth-token your-secret-token

# 2. 访问时携带令牌
curl -H "X-Auth-Token: your-secret-token" http://server:8080/api/data
```

浏览器访问时，HTML 页面无需认证，但 `/api/data` 接口需要 `X-Auth-Token` 头。

> **安全建议**：生产环境建议使用 Nginx 反向代理 + TLS 加密。

### Q7: dashboard 端口被占用怎么办？

```bash
# 方法1：指定其他端口
self-evolving-agent dashboard --port 9090

# 方法2：自动寻找可用端口
self-evolving-agent dashboard --auto-port
```

## 数据清理相关

### Q8: cleanup --dry-run 与实际执行有何区别？

`--dry-run` 模式下：
- 会扫描并列出所有将被清理的文件
- 显示将释放的空间大小
- **不会实际删除任何文件**
- 数据库 VACUUM 也不会实际执行

```bash
# 预览清理效果
self-evolving-agent cleanup --dry-run

# 确认后实际执行
self-evolving-agent cleanup
```

### Q9: 如何只清理快照不清理日志？

```bash
# 只清理快照（保留最近10个）
self-evolving-agent cleanup --keep-snapshots 10

# 只清理日志（保留最近3个轮转文件）
self-evolving-agent cleanup --keep-logs 3

# 只清理数据库碎片
self-evolving-agent cleanup --vacuum-db
```

## 性能相关

### Q10: benchmark 的 --warmup 作用是什么？

预热周期（warmup）在基准测试前运行，**不计入统计**。作用：
- 消除冷启动影响（SQLite 预热、缓存填充、JIT 编译）
- 让预测器和优化器进入稳定状态
- 确保基准测试结果反映稳态性能

```bash
# 推荐：3-5个预热周期
self-evolving-agent benchmark --cycles 50 --warmup 5
```

### Q11: 如何用 JSON 格式导出 benchmark 结果？

```bash
self-evolving-agent --json benchmark --cycles 50 --warmup 5 > benchmark_result.json
```

JSON 输出包含完整的 `config`、`summary`（延迟/吞吐量）、`trends`（奖励/MAE/内存趋势）数据，便于自动化分析。

## 环境变量相关

### Q12: AGENT_ENV 环境变量如何使用？

`AGENT_ENV` 控制默认加载的配置文件：

```bash
# 开发环境（使用 config/config.dev.yaml）
AGENT_ENV=dev self-evolving-agent start

# 测试环境（使用 config/config.test.yaml）
AGENT_ENV=test self-evolving-agent cycle

# 生产环境（默认，使用 config/config.prod.yaml）
self-evolving-agent start
```

| AGENT_ENV | 配置文件 | 日志级别 |
|-----------|----------|----------|
| dev | config/config.dev.yaml | DEBUG |
| test | config/config.test.yaml | WARNING |
| prod（默认） | config/config.prod.yaml | INFO |

### Q13: NO_COLOR 环境变量如何禁用彩色输出？

```bash
# 方法1：环境变量（遵循 NO_COLOR 标准）
NO_COLOR=1 self-evolving-agent status

# 方法2：命令行选项
self-evolving-agent --no-color status
```

两者效果相同，都会禁用终端彩色输出，便于管道处理或日志记录。

## 运行模式相关

### Q14: 如何让 Agent 持续运行？

```bash
# 运行指定周期数
self-evolving-agent start --cycles 100

# 持续运行直到手动停止（Ctrl+C）
self-evolving-agent start

# 后台运行（守护进程模式）
self-evolving-agent start &
```

### Q15: 如何查看 Agent 运行状态？

```bash
# 文本格式
self-evolving-agent status

# JSON 格式（便于脚本处理）
self-evolving-agent --json status

# 查看最近日志
self-evolving-agent logs --lines 50
```

### Q16: 配置热加载如何工作？

Agent 运行时，`config_reloader` 通过 watchdog 监控配置文件变更，自动重新加载配置（无需重启）。支持热加载的配置项：
- `resource_limits`（CPU/内存上限）
- `safety`（安全阈值）
- `logging.level`（日志级别）

不支持热加载的配置项（需重启）：
- `paths`（数据/日志/快照目录）
- `predictors`（预测器类型）

---

## 更多文档

- [故障排查手册](TROUBLESHOOTING.md)
- [安全实践](SECURITY.md)
- [插件接入指南](PLUGIN_GUIDE.md)
- [项目白皮书](PROJECT_WHITEPAPER.md)
