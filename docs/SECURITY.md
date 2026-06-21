# 安全实践

本文档描述 Self-Evolving Agent 的安全模型、配置建议和最佳实践。

## 1. 安全模型概述

Self-Evolving Agent 采用多层安全机制：

```
┌─────────────────────────────────────────────┐
│              行为安全层                       │
│  BehaviorGuardian: 路径检查 + 操作白名单      │
├─────────────────────────────────────────────┤
│              资源安全层                       │
│  ResourceGuardian: CPU/内存/磁盘上限          │
├─────────────────────────────────────────────┤
│              驱力安全层                       │
│  RewardEngine: safe 驱力抑制危险行为          │
├─────────────────────────────────────────────┤
│              紧急冻结层                       │
│  SafetyEnforcer: 连续失败触发紧急模式         │
└─────────────────────────────────────────────┘
```

### 1.1 安全驱力（Reward_safe）

`base_weights.safe` 权重控制安全驱力对总奖励的贡献。当 Agent 执行危险操作（如访问禁止路径、资源超限）时，安全驱力会产生负奖励，抑制此类行为。

```yaml
base_weights:
  safe: 0.4   # 安全驱力权重（越高越保守）
  gain: 0.4   # 收益驱力权重
  eff: 0.2    # 效率驱力权重
```

### 1.2 资源约束

```yaml
resource_limits:
  max_cpu_percent: 20    # CPU 使用上限
  max_mem_percent: 20    # 内存使用上限
  max_db_size_mb: 500    # 数据库大小上限
  max_snapshot_count: 10 # 快照数量上限
```

### 1.3 紧急冻结机制

当连续失败次数超过阈值时，Agent 进入紧急模式，停止探索行为：

```yaml
safety:
  max_consecutive_failures: 5  # 连续失败上限
```

## 2. 路径安全

### 2.1 forbidden_paths 配置

`forbidden_paths` 定义 Agent 不可访问的路径列表。`BehaviorGuardian` 在执行任何文件操作前检查目标路径是否在禁止列表中。

**默认禁止路径**（三套环境配置一致）：

```yaml
safety:
  forbidden_paths:
    - /boot
    - /etc          # 包含 /etc/shadow、/etc/passwd 等敏感文件
    - /proc         # 进程信息
    - /sys          # 系统设备
    - /dev          # 设备文件
    - /bin          # 系统二进制
    - /sbin         # 系统管理二进制
    - /lib          # 系统库
    - /lib64        # 64位系统库
    - /usr/bin      # 用户命令
    - /usr/sbin     # 系统管理命令
```

### 2.2 自定义禁止路径

根据部署环境添加额外的禁止路径：

```yaml
safety:
  forbidden_paths:
    # 保留默认禁止路径
    - /etc
    - /proc
    - /sys
    # 添加自定义禁止路径
    - /home/user/.ssh        # SSH 密钥
    - /var/lib/mysql         # 数据库文件
    - /opt/secrets           # 应用密钥目录
```

### 2.3 路径检查机制

`BehaviorGuardian` 使用前缀匹配检查路径：

```python
# 任何以 /etc 开头的路径都被禁止
forbidden: /etc
访问 /etc/shadow  → 禁止 ✓
访问 /etc/nginx/nginx.conf → 禁止 ✓
访问 /etc-passwd  → 允许（不是 /etc/ 前缀）
```

> **注意**：`forbidden_paths` 中的路径作为前缀匹配，请确保路径以 `/` 结尾或使用完整路径。

## 3. 仪表盘安全

### 3.1 默认绑定 127.0.0.1

Dashboard 默认绑定 `127.0.0.1`，仅允许本机访问。这是安全默认值——仪表盘数据包含 Agent 运行状态、配置路径、日志目录等敏感信息。

### 3.2 --auth-token 鉴权

如需远程访问，必须设置 `--auth-token`：

```bash
# 启动带认证的仪表盘
self-evolving-agent dashboard --host 0.0.0.0 --auth-token $(openssl rand -hex 32)
```

**认证机制**：
- HTML 页面（`/`）：无需认证（不含敏感数据，仅前端框架）
- 数据接口（`/api/data`）：需要 `X-Auth-Token` 请求头
- 健康检查（`/api/health`）：无需认证（仅返回状态码）

**访问示例**：

```bash
# 未认证访问（返回 401）
curl http://server:8080/api/data
# {"error": "Unauthorized"}

# 认证访问
curl -H "X-Auth-Token: your-token" http://server:8080/api/data
# {"cycle_count": 100, "recent_mae": 0.05, ...}
```

### 3.3 CORS 策略

Standalone Dashboard（CLI 启动的仪表盘）**不设置任何 CORS 头**，依赖网络隔离防止跨域访问。

如需通过浏览器前端访问 API，建议使用 Nginx 反向代理并配置 CORS：

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_set_header X-Auth-Token $http_x_auth_token;
    add_header Access-Control-Allow-Origin "https://your-frontend.com" always;
    add_header Access-Control-Allow-Headers "X-Auth-Token" always;
}
```

### 3.4 远程访问安全建议

| 措施 | 优先级 | 说明 |
|------|--------|------|
| 设置 `--auth-token` | 必须 | 防止未授权访问 |
| 使用 TLS 加密 | 推荐 | 通过 Nginx/Caddy 反向代理实现 HTTPS |
| 限制访问 IP | 推荐 | 防火墙规则限制来源 IP |
| 不暴露到公网 | 推荐 | 使用 VPN 或 SSH 隧道访问 |

## 4. 数据安全

### 4.1 数据目录权限

```bash
# 设置目录权限（仅所有者可读写执行）
chmod 700 data/ snapshots/
chmod 750 logs/

# 确保目录所有者正确
chown $USER:$USER data/ logs/ snapshots/
```

### 4.2 敏感数据位置

| 数据 | 位置 | 敏感度 | 说明 |
|------|------|--------|------|
| memory.db | data/ | 中 | Agent 记忆数据，含运行历史 |
| metrics.json | data/ | 低 | 运行指标，无敏感信息 |
| agent.pid | data/ | 低 | PID 文件 |
| agent.log | logs/ | 中 | 运行日志，可能含路径信息 |
| snapshots/*.pkl | snapshots/ | 中 | 模型快照 |
| config.yaml | config/ | 中 | 配置文件，含路径但不含密钥 |

### 4.3 配置文件安全

配置文件不含密码、API 密钥等高敏感信息，但包含路径和资源限制。建议：

```bash
# 限制配置文件访问权限
chmod 600 config/config.prod.yaml
```

## 5. 运行时安全

### 5.1 紧急冻结机制

当 Agent 连续失败超过阈值时，自动进入紧急模式：

```yaml
safety:
  max_consecutive_failures: 5  # 连续失败上限
```

紧急模式下：
- Agent 停止探索行为（exploration_ratio 降至 0）
- 仅执行安全任务
- 日志记录 `Emergency mode activated`
- 需要连续成功若干次才能退出紧急模式

### 5.2 模型重构限制

```yaml
optimization:
  max_model_restructure_iterations: 3  # 单周期最大重构次数
```

防止 Agent 在单周期内过度修改自身模型导致不稳定。

### 5.3 探索止损

```yaml
exploration:
  base_ratio: 0.3       # 基础探索率
  stable_threshold: 0.05 # 稳定阈值
  emergency_ratio: 0.1   # 紧急模式探索率
  novel_ratio: 0.4       # 新颖性探索率
```

探索率受安全驱力调节，当安全驱力低时自动降低探索率。

## 6. 安全审计

### 6.1 日志审计

启用 JSON 格式日志便于审计分析：

```yaml
logging:
  format_type: json
  level: INFO
```

JSON 日志每行包含 `timestamp`、`level`、`logger`、`message`、`module`、`func`、`line` 等字段，可通过 `jq` 等工具分析：

```bash
# 查找所有安全相关日志
cat logs/agent.log | jq 'select(.logger | contains("safety"))'

# 查找所有 ERROR 级别日志
cat logs/agent.log | jq 'select(.level == "ERROR")'

# 统计各模块日志数量
cat logs/agent.log | jq -r '.logger' | sort | uniq -c | sort -rn
```

### 6.2 健康检查

```bash
# 定期运行健康检查
self-evolving-agent doctor

# JSON 格式输出（便于集成监控系统）
self-evolving-agent --json doctor
```

### 6.3 配置校验

```bash
# 部署前校验配置
self-evolving-agent config validate

# JSON 格式输出
self-evolving-agent --json config validate
```

### 6.4 按模块级别审计日志

```yaml
logging:
  level: INFO
  loggers:
    src.core.safety_enforcer: DEBUG     # 安全执行器详细日志
    src.core.behavior_guardian: DEBUG   # 行为护栏详细日志
    src.core.resource_guardian: DEBUG   # 资源护栏详细日志
```

## 7. 安全检查清单

部署前请逐项确认：

- [ ] `forbidden_paths` 包含 `/etc`、`/proc`、`/sys` 等系统路径
- [ ] `resource_limits` 的 CPU/内存上限符合环境要求
- [ ] `max_consecutive_failures` 设置合理（默认5）
- [ ] Dashboard 绑定 `127.0.0.1` 或设置了 `--auth-token`
- [ ] 数据目录权限设置为 `700` 或 `750`
- [ ] 配置文件权限设置为 `600`
- [ ] 日志级别不低于 `INFO`
- [ ] `config validate` 通过
- [ ] `doctor` 健康检查全部通过

---

## 相关文档

- [故障排查手册](TROUBLESHOOTING.md)
- [常见问题](FAQ.md)
- [插件接入指南](PLUGIN_GUIDE.md)
