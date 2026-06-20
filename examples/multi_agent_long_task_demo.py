#!/usr/bin/env python3
"""自驱动多 Agent 长任务端到端示例。

演示三大应用场景：
1. 自主学会操作多个 agent 完成任务
2. 自主学会增强自己的技能
3. 自主学会自己驱动自己完成长任务

运行方式：
    python examples/multi_agent_long_task_demo.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.long_term import LongTermMemory
from src.multi_agent.agent_worker import AgentWorker
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.planning.long_task_planner import LongTaskPlanner
from src.planning.long_task_reviewer import LongTaskReviewer
from src.planning.multi_agent_runner import MultiAgentLongTaskRunner
from src.skills.skill import Skill
from src.skills.skill_learner import SkillLearner
from src.skills.skill_registry import SkillRegistry
from src.task.task_definition import Task


def build_orchestrator():
    """构建多 Agent 编排器，注册具备不同能力的 worker。"""
    orch = MultiAgentOrchestrator(strategy='capability_match')
    orch.register_worker(AgentWorker(worker_id='collector', capabilities=['network', 'cpu']))
    orch.register_worker(AgentWorker(worker_id='processor', capabilities=['cpu', 'memory']))
    orch.register_worker(AgentWorker(worker_id='storage', capabilities=['disk']))
    return orch


def register_skills(runner):
    """向 runner 注册数据处理流水线技能。"""
    runner.register_skill(
        'fetch_data',
        lambda t: f"fetched {t.params.get('source', 'unknown')}",
        ['network'],
        '从网络源获取数据',
    )
    runner.register_skill(
        'normalize',
        lambda t: f"normalized {t.params.get('records', 0)} records",
        ['cpu'],
        '数据标准化处理',
    )
    runner.register_skill(
        'validate',
        lambda t: f"validated {t.params.get('records', 0)} records",
        ['memory'],
        '数据校验',
    )
    runner.register_skill(
        'persist',
        lambda t: f"persisted to {t.params.get('path', '/data')}",
        ['disk'],
        '持久化存储',
    )


def run_pipeline_demo():
    """场景 1+3：多 Agent 协作完成长任务流水线。"""
    print("\n" + "=" * 60)
    print("场景 1+3：多 Agent 协作自驱动长任务流水线")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        memory = LongTermMemory(base_dir=os.path.join(tmp, 'ltm'))
        orchestrator = build_orchestrator()
        registry = SkillRegistry()
        reviewer = LongTaskReviewer(memory, registry)
        runner = MultiAgentLongTaskRunner(
            orchestrator,
            registry=registry,
            memory=memory,
            reviewer=reviewer,
        )
        register_skills(runner)

        goal = Task(name='data_pipeline', description='端到端数据处理流水线')
        steps = [
            Task(name='fetch_data', params={'required_capabilities': ['network'], 'source': 'api.example.com'}),
            Task(name='normalize', params={'required_capabilities': ['cpu'], 'records': 1000}),
            Task(name='validate', params={'required_capabilities': ['memory'], 'records': 1000}),
            Task(name='persist', params={'required_capabilities': ['disk'], 'path': '/data/output'}),
        ]

        summary = runner.run(goal, steps, auto_review=True)
        print(f"\n执行状态: {summary['status']}")
        print(f"总步骤: {summary['total_steps']}, 成功: {summary['completed_steps']}, 失败: {summary['failed_steps']}")
        for r in summary['results']:
            status = '✓' if r['success'] else '✗'
            print(f"  {status} [{r['worker_id']}] {r['task_name']} -> {r['result']}")

        if 'review' in summary:
            review = summary['review']
            print(f"\n复盘报告 (review_id={review['review_id']}):")
            print(f"  成功率: {review['success_rate']:.1%}")
            print(f"  技能建议: {review['skill_recommendations']}")

        perf = runner.performance_summary()
        print("\n性能指标:")
        for op, stats in perf.items():
            print(f"  {op}: count={stats['count']}, mean={stats['mean']:.4f}s, p95={stats['p95']:.4f}s")

        return runner


def run_skill_enhancement_demo():
    """场景 2：技能自增强——从失败中学习并优化。"""
    print("\n" + "=" * 60)
    print("场景 2：技能自增强与反馈学习")
    print("=" * 60)

    registry = SkillRegistry()
    learner = SkillLearner(registry)

    # 注册一个不稳定技能
    call_state = {'count': 0}

    def flaky_action(task):
        call_state['count'] += 1
        if call_state['count'] < 3:
            raise RuntimeError(f"transient failure #{call_state['count']}")
        return f"recovered after {call_state['count']} attempts"

    skill = Skill(name='flaky_skill', action=flaky_action, capabilities=['cpu'])
    registry.register(skill)

    # 模拟多次执行并吸收反馈
    task = Task(name='flaky_skill', params={'required_capabilities': ['cpu']})
    for i in range(5):
        result = skill.execute(task)
        learner.feedback('flaky_skill', result['success'])
        status = '✓' if result['success'] else '✗'
        print(f"  执行 #{i + 1}: {status} -> {result.get('result')}")

    recommendations = learner.improve()
    print(f"\n技能统计: success={skill.success_count}, failure={skill.failure_count}, rate={skill.success_rate:.1%}")
    print(f"优化建议: {recommendations}")

    return registry


def run_long_task_self_driven_demo():
    """场景 3：长任务自驱动规划与断点续跑。"""
    print("\n" + "=" * 60)
    print("场景 3：长任务自驱动规划与 checkpoint 恢复")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        memory = LongTermMemory(base_dir=os.path.join(tmp, 'ltm'))
        goal = Task(name='batch_processing', description='批量数据处理长任务')
        planner = LongTaskPlanner(goal, memory=memory, max_steps=10)

        for i in range(5):
            planner.add_step(
                Task(name=f'batch_{i}', params={'batch_id': i}),
                action=lambda t: f"processed batch {t.params['batch_id']}",
            )

        summary = planner.run()
        print(f"\n首次执行: status={summary['status']}, completed={summary['completed_steps']}")
        checkpoint_path = planner.checkpoint()
        print(f"Checkpoint 已保存: {checkpoint_path}")

        # 模拟新实例恢复
        planner2 = LongTaskPlanner(goal, memory=memory, max_steps=10)
        for i in range(5):
            planner2.add_step(
                Task(name=f'batch_{i}', params={'batch_id': i}),
                action=lambda t: f"reprocessed batch {t.params['batch_id']}",
            )
        planner2.plan_id = planner.plan_id
        resumed = planner2.resume()
        if resumed:
            print(f"恢复执行: status={resumed['status']}, completed={resumed['completed_steps']}")


def main():
    print("=" * 60)
    print("自驱永续进化 Agent - 多 Agent 长任务端到端示例")
    print("=" * 60)

    run_pipeline_demo()
    run_skill_enhancement_demo()
    run_long_task_self_driven_demo()

    print("\n" + "=" * 60)
    print("所有场景演示完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
