#!/usr/bin/env python3
"""独立启动仪表盘预览服务器，带模拟数据"""

import os
import sys
import random
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.server import DashboardServer
from src.monitoring.alert_manager import AlertManager


def generate_mock_data(store):
    """持续生成模拟周期数据"""
    cycle = 0
    history = []
    while True:
        cycle += 1
        mae = max(0.01, 0.1 + random.uniform(-0.05, 0.05))
        reward = max(0.1, 0.6 + random.uniform(-0.2, 0.2))
        history.append({'cycle': cycle, 'mae': mae, 'reward': reward, 'duration': 0.01})
        if len(history) > 100:
            history = history[-100:]

        result = {
            'cycle': cycle,
            'state': {'cpu': random.uniform(5, 30), 'mem': random.uniform(20, 60)},
            'predictions': {
                'low': {'cpu': random.uniform(5, 30)},
                'mid': {'cpu': random.uniform(5, 30)},
                'high': {'trend': 'stable'},
            },
            'errors': {'cpu': mae},
            'reward': {'total_reward': reward, 'safe_reward': 0.3, 'gain_reward': 0.2, 'eff_reward': 0.1},
            'execution': {'status': 'executed'},
            'safety': {'emergency_mode': False, 'resource_safe': True},
            'duration_seconds': 0.01,
        }
        store.update(result)
        store.update_stats({
            'recent_mae': mae,
            'recent_success_rate': random.uniform(0.6, 0.95),
            'task_pool': {'total_tasks': random.randint(5, 20), 'active_tasks': random.randint(3, 10)},
            'memory_samples': min(cycle * 2, 1000),
        })
        time.sleep(1.0)


def inject_mock_alerts(alert_manager):
    """注入模拟告警"""
    time.sleep(3)
    alert_manager.trigger('warning', 'system', 'CPU usage spike detected')
    time.sleep(5)
    alert_manager.trigger('info', 'task_pool', 'New exploration task generated')
    time.sleep(4)
    alert_manager.trigger('warning', 'memory', 'Short-term memory approaching limit')
    time.sleep(6)
    alert_manager.trigger('critical', 'safety', 'Emergency mode triggered: resource violation')


def main():
    host = '0.0.0.0'
    port = 8080
    server = DashboardServer(host=host, port=port)
    alert_manager = AlertManager()
    server.set_alert_manager(alert_manager)
    server.start()

    print(f"Dashboard preview started at http://{host}:{port}")
    print("Open your browser to view the dashboard.")
    print("Press Ctrl+C to stop.")

    t1 = threading.Thread(target=generate_mock_data, args=(server.data_store,), daemon=True)
    t2 = threading.Thread(target=inject_mock_alerts, args=(alert_manager,), daemon=True)
    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == '__main__':
    main()
