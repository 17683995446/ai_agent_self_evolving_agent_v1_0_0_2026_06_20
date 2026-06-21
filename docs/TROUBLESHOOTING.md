# 故障排查手册

本手册汇总 Self-Evolving Agent 常见问题的诊断与解决方法。

## 1. 启动问题

### 1.1 配置文件未找到

**现象**：运行命令时报错 `Config file not found: config/config.prod.yaml`

**解决**：

```bash
# 方法1：生成默认配置
self-evolving-agent config init

# 方法2：交互式向导生成定制配置
self-evolving-agent config wizard

# 方法3：一键快速体验（自动初始化配置）
self-evolving-agent quickstart

# 方法4：指定配置文件路径
self-evolving-agent -c /path/to/config.yaml start
```

### 1.2 配置校验失败

**现象**：运行 `config validate` 报错

**常见错误**：

| 错误信息 | 原因 | 解决 |
|----------|------|------|
| `base_weights.safe+gain+eff must sum to 1.0` | 驱力权重和不为1 | 调整 `base_weights` 三项值使其和为1.0 |
| `resource_limits.max_cpu_percent must be 0~100` | CPU上限越界 | 设置为 0-100 之间的数值 |
| `logging.level must be one of {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | 日志级别非法 | 使用合法的级别名称 |
| `logging.format_type must be 'text' or 'json'` | 格式类型非法 | 设置为 `text` 或 `json` |
| `paths.data_dir must be a string` | 字段类型错误 | 确保路径为字符串类型 |

**批量修复**：

```bash
# 重新生成默认配置（覆盖）
self-evolving-agent config init --force
```

### 1.3 AgentLoop 初始化失败

**诊断步骤**：

```bash
# 1. 校验配置
self-evolving-agent config validate

# 2. 健康检查
self-evolving-agent doctor

# 3. 查看日志
self-evolving-agent logs --lines 50
```

## 2. 运行时问题

### 2.1 PID 文件残留导致 stop 无效

**现象**：`status` 显示 Agent 正在运行，但实际进程已退出

**原因**：Agent 异常退出未清理 PID 文件

**解决**：

```bash
# 检查进程是否真的存在
ps aux | grep self-evolving

# 如果进程不存在，手动清理 PID 文件
rm data/agent.pid

# 重新启动
self-evolving-agent start
```

### 2.2 连续失败触发紧急模式

**现象**：日志中出现 `Emergency mode activated`，Agent 停止探索

**原因**：连续失败次数超过 `safety.max_consecutive_failures`（默认5次）

**解决**：

```yaml
# 调整配置中的阈值
safety:
  max_consecutive_failures: 10  # 提高容忍度
```

```bash
# 重启 Agent
self-evolving-agent stop && self-evolving-agent start
```

### 2.3 资源超限

**现象**：日志中出现 `CPU usage exceeded limit` 或 `Memory usage exceeded limit`

**解决**：

```yaml
# 调整资源限制
resource_limits:
  max_cpu_percent: 30  # 从20提高到30
  max_mem_percent: 30  # 从20提高到30
```

## 3. 数据库问题

### 3.1 SQLite database is locked

**现象**：日志中出现 `sqlite3.OperationalError: database is locked`

**原因**：多个进程同时写入 memory.db

**解决**：

```bash
# 确保只有一个 Agent 实例在运行
self-evolving-agent status
self-evolving-agent stop

# 清理并重启
self-evolving-agent cleanup --vacuum-db
self-evolving-agent start
```

**预防**：项目已启用 WAL 模式和 10 秒超时，正常使用不应出现此问题。

### 3.2 数据库体积过大

**诊断**：

```bash
# 查看数据库大小
ls -lh data/memory.db

# 清理碎片空间
self-evolving-agent cleanup --vacuum-db

# 预览清理效果（不实际执行）
self-evolving-agent cleanup --vacuum-db --dry-run
```

**配置调整**：

```yaml
resource_limits:
  max_db_size_mb: 200  # 降低上限触发自动清理
```

## 4. 路径与权限问题

### 4.1 数据目录不可写

**现象**：`doctor` 检查报 `Path: data: not writable`

**解决**：

```bash
# 检查目录权限
ls -ld data/ logs/ snapshots/

# 修改权限
chmod 755 data/ logs/ snapshots/
chown $USER:$USER data/ logs/ snapshots/
```

### 4.2 快照目录不存在

**现象**：日志中出现 `Snapshot directory not found`

**解决**：

```bash
# 手动创建
mkdir -p snapshots

# 或通过 doctor 自动检测
self-evolving-agent doctor
```

## 5. 仪表盘问题

### 5.1 端口被占用

**现象**：`Failed to start dashboard on 127.0.0.1:8080`

**解决**：

```bash
# 方法1：使用其他端口
self-evolving-agent dashboard --port 9090

# 方法2：自动寻找可用端口
self-evolving-agent dashboard --auto-port

# 方法3：查看占用端口的进程
lsof -i :8080
```

### 5.2 远程访问被拒

**现象**：从其他机器访问仪表盘无法连接

**原因**：默认绑定 `127.0.0.1`，仅允许本机访问

**解决**（务必配合认证）：

```bash
# 绑定所有网卡 + 设置认证令牌
self-evolving-agent dashboard --host 0.0.0.0 --auth-token your-secret-token

# 访问时携带令牌
curl -H "X-Auth-Token: your-secret-token" http://server:8080/api/data
```

> **安全警告**：远程访问务必设置 `--auth-token`，否则任何人都能读取 Agent 数据。

### 5.3 仪表盘无数据

**现象**：仪表盘页面打开但无数据显示

**诊断**：

```bash
# 1. 检查 Agent 是否在运行
self-evolving-agent status

# 2. 检查 metrics.json 是否存在
ls -la data/metrics.json

# 3. 手动运行一个周期生成数据
self-evolving-agent cycle
```

## 6. 日志问题

### 6.1 日志文件未找到

**现象**：`logs` 命令报 `Log file not found`

**解决**：

```bash
# 检查日志目录
ls -la logs/

# 如果目录为空，运行一个周期生成日志
self-evolving-agent cycle

# 指定日志文件路径
self-evolving-agent logs --lines 100
```

### 6.2 切换 JSON 格式日志

**配置修改**：

```yaml
logging:
  format_type: json  # 从 text 改为 json
```

**验证**：

```bash
self-evolving-agent cycle
# 查看日志，每行应为 JSON 格式
self-evolving-agent logs --lines 5
```

### 6.3 日志体积过大

**解决**：

```bash
# 清理旧日志
self-evolving-agent cleanup --keep-logs 3

# 调整配置
```

```yaml
logging:
  max_bytes: 5242880    # 5MB（默认10MB）
  backup_count: 3       # 保留3个轮转文件（默认5）
```

## 7. 依赖问题

### 7.1 依赖导入失败

**诊断**：

```bash
# 健康检查会检测所有依赖
self-evolving-agent doctor
```

**解决**：

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或重新安装项目（含依赖）
pip install -e .
```

### 7.2 Python 版本过低

**要求**：Python 3.10+

**检查**：

```bash
python --version
# 或
self-evolving-agent doctor
```

## 8. 性能问题

### 8.1 周期执行缓慢

**诊断**：

```bash
# 运行性能基准测试
self-evolving-agent benchmark --cycles 20 --warmup 3

# 查看延迟分布
self-evolving-agent --json benchmark --cycles 50 --warmup 5
```

**参考基线**：单周期延迟应 < 500ms（参见 `tests/test_stage127_performance_baseline.py`）

### 8.2 内存持续增长

**诊断**：

```bash
# 运行长时间 benchmark 观察内存趋势
self-evolving-agent benchmark --cycles 100 --warmup 5
# 观察"内存 RSS"行，如果 delta 持续增长可能存在泄漏
```

**解决**：

```bash
# 清理记忆数据库
self-evolving-agent cleanup --vacuum-db

# 降低记忆保留量
```

```yaml
memory:
  max_short_term_samples: 5000  # 从10000降低
  value_retention_ratio: 0.2   # 从0.3降低
```

---

## 获取更多帮助

- **完整命令列表**：`self-evolving-agent --help`
- **健康检查**：`self-evolving-agent doctor`
- **配置校验**：`self-evolving-agent config validate`
- **项目文档**：`docs/` 目录
