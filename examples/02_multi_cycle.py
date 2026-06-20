#!/usr/bin/env python3
"""示例2：多周期连续运行与状态监控。

演示如何连续运行多个周期，并在运行过程中监控 Agent 状态。
适合想了解系统持续运行行为的用户。

运行方式：
    python examples/02_multi_cycle.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.agent_loop import AgentLoop


def main():
    config_path = 'config/config.prod.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return 1

    agent = AgentLoop(config_path)

    try:
        num_cycles = 5
        print(f"Running {num_cycles} cycles...\n")

        for i in range(num_cycles):
            result = agent.run_cycle()
            stats = agent.get_stats()

            print(f"[Cycle {result['cycle']}] "
                  f"reward={result['reward']['total_reward']:.4f}, "
                  f"mae={stats['recent_mae']:.4f}, "
                  f"samples={stats['memory_samples']}")

            time.sleep(0.5)

        print(f"\nCompleted {num_cycles} cycles.")
        print("\nFinal stats:")
        final_stats = agent.get_stats()
        print(f"  Total cycles: {final_stats['cycle_count']}")
        print(f"  Memory samples: {final_stats['memory_samples']}")
        print(f"  Recent MAE: {final_stats['recent_mae']:.4f}")
        print(f"  Success rate: {final_stats['recent_success_rate']:.2%}")
        return 0
    finally:
        agent.stop()


if __name__ == '__main__':
    sys.exit(main())
