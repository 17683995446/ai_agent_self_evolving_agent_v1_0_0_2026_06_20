#!/usr/bin/env python3
"""感知层性能分析"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perception.system_sensor import SystemSensor
from src.perception.log_sensor import LogSensor
from src.perception.file_sensor import FileSensor


def profile_sensors():
    sensors = {
        'system': SystemSensor(),
        'log': LogSensor(),
        'file': FileSensor(),
    }

    # 预热
    for name, sensor in sensors.items():
        sensor.perceive()

    phases = {}
    for name, sensor in sensors.items():
        start = time.time()
        for _ in range(10):
            sensor.perceive()
        duration = time.time() - start
        phases[name] = duration / 10

    print("\n" + "=" * 60)
    print("感知层性能分析（单次耗时）")
    print("=" * 60)
    total = sum(phases.values())
    for name, duration in sorted(phases.items(), key=lambda x: -x[1]):
        pct = (duration / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{name:20s} {duration:8.4f}s ({pct:5.1f}%) {bar}")
    print("-" * 60)
    print(f"{'TOTAL':20s} {total:8.4f}s")
    print("=" * 60)


if __name__ == '__main__':
    profile_sensors()
