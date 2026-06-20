#!/usr/bin/env python3
"""示例1：最简单的单周期运行。

演示如何用最少的代码启动 Agent 并执行一个完整周期。
适合第一次接触本项目的用户。

运行方式：
    python examples/01_single_cycle.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop


def main():
    # 使用默认的生产配置
    config_path = 'config/config.prod.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Run 'self-evolving-agent config init' first.")
        return 1

    # 创建 Agent 实例
    agent = AgentLoop(config_path)

    try:
        # 执行单个周期
        result = agent.run_cycle()

        # 打印关键结果
        print(f"Cycle {result['cycle']} completed in {result['duration_seconds']:.4f}s")
        print(f"  Reward: {result['reward']['total_reward']:.4f}")
        print(f"  Execution: {result['execution']['status']}")
        print(f"  Emergency: {result['safety']['emergency_mode']}")
        return 0
    finally:
        agent.stop()


if __name__ == '__main__':
    sys.exit(main())
