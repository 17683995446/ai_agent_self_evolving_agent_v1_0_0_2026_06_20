#!/usr/bin/env python3
"""性能分析：定位每个模块的执行时间瓶颈"""

import os
import sys
import tempfile
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import Config


def profile_agent_cycle():
    Config._instance = None
    Config._config = {}

    temp_dir = tempfile.mkdtemp()
    orig_cwd = os.getcwd()
    os.chdir(temp_dir)

    for d in ['data', 'logs', 'snapshots']:
        os.makedirs(d, exist_ok=True)

    with open('config.yaml', 'w') as f:
        f.write("""
goals:
  precision_evolution: true
base_weights:
  safe: 0.4
  gain: 0.4
  eff: 0.2
resource_limits:
  max_cpu_percent: 100.0
  max_mem_percent: 100.0
exploration:
  base_ratio: 0.3
safety:
  max_consecutive_failures: 5
review:
  interval_hours: 6
  min_samples: 100
memory:
  max_short_term_samples: 100
logging:
  level: WARNING
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/agent.log
paths:
  data_dir: data
  snapshot_dir: snapshots
  log_dir: logs
dashboard:
  enabled: false
""")

    from src.core.agent_loop import AgentLoop
    agent = AgentLoop('config.yaml')

    # 预热
    agent.run_cycle()

    # 详细分析每个步骤
    phases = {}

    def time_phase(name, func):
        start = time.time()
        result = func()
        duration = time.time() - start
        phases[name] = duration
        return result

    # 手动拆解run_cycle的每个步骤
    state = time_phase('perceive', lambda: agent.state_vector.perceive())
    flat_state = agent.state_vector.flatten(state)

    time_phase('safety', lambda: agent.safety_enforcer.full_check())

    time_phase('low_predict', lambda: agent.low_predictor.predict())
    time_phase('mid_predict', lambda: agent.mid_predictor.predict())

    low_preds = agent.low_predictor.predict()
    mid_preds = agent.mid_predictor.predict()
    time_phase('high_predict', lambda: agent.high_predictor.predict(low_preds, mid_preds))

    actuals = {
        'cpu_percent': flat_state.get('system.cpu_percent', 0.0),
        'mem_percent': flat_state.get('system.mem_percent', 0.0),
        'disk_percent': flat_state.get('system.disk_percent', 0.0),
    }
    time_phase('error_eval', lambda: agent.error_evaluator.evaluate(low_preds, actuals))

    time_phase('low_update', lambda: agent.low_predictor.update(actuals))
    time_phase('mid_update', lambda: agent.mid_predictor.update(actuals))

    time_phase('reward', lambda: agent.reward_engine.compute_total_reward(state, []))

    time_phase('task_generate', lambda: agent.task_generator.generate({
        'safe_gap': 0.1, 'gain_gap': 0.1, 'eff_gap': 0.1
    }))

    time_phase('task_pool_update', lambda: agent.task_pool.update([]))
    time_phase('task_select', lambda: agent.task_pool.select_task())

    time_phase('memory_store', lambda: agent.memory.store_sample({
        'timestamp': time.time(), 'state': state, 'predictions': {},
        'actuals': actuals, 'errors': {}, 'task_id': None,
        'reward': 0.5, 'value_score': 0.5,
    }))

    agent.stop()
    os.chdir(orig_cwd)
    shutil.rmtree(temp_dir, ignore_errors=True)

    # 输出结果
    print("\n" + "=" * 60)
    print("性能分析结果（单次周期各阶段耗时）")
    print("=" * 60)
    total = sum(phases.values())
    for name, duration in sorted(phases.items(), key=lambda x: -x[1]):
        pct = (duration / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{name:20s} {duration:8.4f}s ({pct:5.1f}%) {bar}")
    print("-" * 60)
    print(f"{'TOTAL':20s} {total:8.4f}s")
    print("=" * 60)

    return phases


if __name__ == '__main__':
    profile_agent_cycle()
