import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.planning.long_task_planner import LongTaskPlanner
from src.task.task_definition import Task
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.memory.long_term import LongTermMemory


class TestLongTaskPlanner(unittest.TestCase):
    """LongTaskPlanner 单元测试"""

    def test_create_status(self):
        goal = Task(name='deploy_service')
        planner = LongTaskPlanner(goal)
        self.assertEqual(planner.total_steps, 0)
        self.assertEqual(planner.status()['status'], 'pending')

    def test_run_all_steps_with_local_worker(self):
        goal = Task(name='sum_numbers')
        planner = LongTaskPlanner(goal)
        for i in range(3):
            planner.add_step(
                Task(name=f'step_{i}', params={'value': i}),
                action=lambda t, base=i: base * 2,
            )
        summary = planner.run()
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['completed_steps'], 3)
        self.assertEqual(summary['failed_steps'], 0)
        self.assertEqual(len(summary['results']), 3)

    def test_run_with_orchestrator(self):
        orch = MultiAgentOrchestrator()
        orch.register_worker(AgentWorker(worker_id='w1', capabilities=['cpu']))
        goal = Task(name='distributed_sum')
        planner = LongTaskPlanner(goal, orchestrator=orch)
        planner.add_step(
            Task(name='double', params={'value': 5, 'required_capabilities': ['cpu']}),
            action=lambda t: t.params['value'] * 2,
        )
        summary = planner.run()
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['results'][0]['result'], 10)

    def test_failure_stops_by_default(self):
        goal = Task(name='fragile')
        planner = LongTaskPlanner(goal)
        planner.add_step(Task(name='ok'), action=lambda t: 'ok')
        planner.add_step(Task(name='fail'), action=lambda t: 1 / 0)
        planner.add_step(Task(name='never'), action=lambda t: 'never')
        summary = planner.run()
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['completed_steps'], 1)
        self.assertEqual(summary['failed_steps'], 1)

    def test_continue_on_failure(self):
        goal = Task(name='tolerant')
        planner = LongTaskPlanner(goal, continue_on_failure=True)
        planner.add_step(Task(name='ok'), action=lambda t: 'ok')
        planner.add_step(Task(name='fail'), action=lambda t: 1 / 0)
        planner.add_step(Task(name='ok2'), action=lambda t: 'ok2')
        summary = planner.run()
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['completed_steps'], 2)
        self.assertEqual(summary['failed_steps'], 1)

    def test_retry_then_succeed(self):
        goal = Task(name='flaky')
        call_count = {'n': 0}

        def flaky(t):
            call_count['n'] += 1
            if call_count['n'] < 2:
                raise RuntimeError('transient')
            return 'recovered'

        planner = LongTaskPlanner(goal)
        planner.add_step(Task(name='step'), action=flaky, retries=2)
        summary = planner.run()
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['completed_steps'], 1)
        self.assertEqual(call_count['n'], 2)

    def test_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(base_dir=os.path.join(tmp, 'ltm'))
            goal = Task(name='resume_test')
            planner = LongTaskPlanner(goal, memory=memory)
            for i in range(3):
                planner.add_step(
                    Task(name=f'step_{i}', params={'value': i}),
                    action=lambda t: t.params['value'],
                )
            # 执行前两步后 checkpoint
            planner.run()
            planner.checkpoint()

            # 新 planner 实例恢复并继续
            planner2 = LongTaskPlanner(goal, memory=memory)
            # 需要重新添加步骤，因为 resume 依赖 checkpoint 中的 step_index
            for i in range(3):
                planner2.add_step(
                    Task(name=f'step_{i}', params={'value': i}),
                    action=lambda t: t.params['value'],
                )
            planner2.plan_id = planner.plan_id
            summary = planner2.resume()
            self.assertIsNotNone(summary)
            self.assertEqual(summary['status'], 'completed')
            # 恢复后应只执行剩余步骤
            self.assertEqual(summary['completed_steps'], 3)

    def test_replan_callback(self):
        goal = Task(name='adaptive')
        planner = LongTaskPlanner(
            goal,
            replan_callback=lambda p, idx, record: [
                Task(name='fallback', params={'value': 42}),
                Task(name='after'),
            ],
        )
        planner.add_step(Task(name='fail'), action=lambda t: 1 / 0, retries=0)
        planner.add_step(Task(name='old_after'), action=lambda t: 'old_after')
        summary = planner.run()
        # replan 替换了失败点之后的计划，fallback 与新的 after 都应执行
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['total_steps'], 3)
        self.assertEqual(summary['completed_steps'], 2)

    def test_max_steps_limit(self):
        goal = Task(name='infinite')
        planner = LongTaskPlanner(goal, max_steps=2)
        for i in range(5):
            planner.add_step(Task(name=f'step_{i}'), action=lambda t: 'ok')
        summary = planner.run()
        self.assertEqual(summary['status'], 'paused')
        self.assertEqual(len(summary['results']), 2)


if __name__ == '__main__':
    unittest.main()
