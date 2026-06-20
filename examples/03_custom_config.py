#!/usr/bin/env python3
"""示例3：自定义配置运行。

演示如何使用自定义配置文件运行 Agent，调整资源限制和探索策略。
适合想根据自己需求调整系统行为的用户。

运行方式：
    python examples/03_custom_config.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop


def create_custom_config():
    """创建一个自定义配置文件，使用更保守的资源限制。"""
    config_content = """
goals:
  precision_evolution: true
  coverage_evolution: true
  efficiency_evolution: true

base_weights:
  safe: 0.5
  gain: 0.3
  eff: 0.2

resource_limits:
  max_cpu_percent: 10.0
  max_mem_percent: 10.0
  max_db_size_mb: 200
  max_snapshot_count: 5

exploration:
  base_ratio: 0.2
  stable_threshold: 0.05
  emergency_ratio: 0.1
  novel_ratio: 0.3

models:
  smoothing_alpha: 0.3
  smoothing_alpha_min: 0.05
  smoothing_alpha_max: 0.8
  arima_order: [1, 1, 1]
  prediction_horizon: 1

safety:
  max_consecutive_failures: 3
  forbidden_paths:
    - /boot
    - /etc
    - /proc
  max_model_restructure_iterations: 3

review:
  interval_hours: 6
  min_samples: 50
  idle_cpu_threshold: 10.0
  idle_duration_seconds: 300

memory:
  max_short_term_samples: 5000
  value_retention_ratio: 0.3
  history_window: 100

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log
  max_bytes: 10485760
  backup_count: 3

paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
"""
    config_path = os.path.join(tempfile.gettempdir(), 'agent_custom_config.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path


def main():
    config_path = create_custom_config()
    print(f"Using custom config: {config_path}")
    print("  - Conservative resource limits (CPU<=10%, memory<=10%)")
    print("  - Higher safety weight (safe=0.5)")
    print("  - Lower exploration ratio (0.2)")
    print()

    agent = AgentLoop(config_path)

    try:
        result = agent.run_cycle()
        print(f"Cycle {result['cycle']} completed")
        print(f"  Reward: {result['reward']['total_reward']:.4f}")
        print(f"  Execution: {result['execution']['status']}")
        return 0
    finally:
        agent.stop()


if __name__ == '__main__':
    sys.exit(main())
