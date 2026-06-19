import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.planning.long_task_planner import LongTaskPlanner
from src.planning.multi_agent_runner import MultiAgentLongTaskRunner
from src.planning.long_task_reviewer import LongTaskReviewer
from src.multi_agent.orchestrator import MultiAgentOrchestrator
from src.multi_agent.agent_worker import AgentWorker
from src.skills.skill import Skill
from src.skills.skill_registry import SkillRegistry
from src.memory.long_term import LongTermMemory
from src.task.task_definition import Task


class TestEmptySkillRegistry(unittest.TestCase):
    """空技能注册表边界"""

    def test_runner_with_no_skills_uses_default_execution(self):
        orch = MultiAgentOrchestrator()
        orch.register_worker(AgentWorker(worker_id='w1', capabilities=['cpu']))
        runner = MultiAgentLongTaskRunner(orch)
        # 无任何技能注册，步骤应通过 worker 默认占位执行
        goal = Task(name='goal')
        steps = [Task(name='step1', params={'required_capabilities': ['cpu']})]
        summary = runner.run(goal, steps)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['completed_steps'], 1)

    def test_select_best_returns_none_on_empty_registry(self):
        registry = SkillRegistry()
        task = Task(name='job', params={'required_capabilities': ['cpu']})
        self.assertIsNone(registry.select_best(task))
        self.assertIsNone(registry.execute(task))


class TestAllWorkersUnavailable(unittest.TestCase):
    """所有 Worker 不可用边界"""

    def test_orchestrator_no_workers_returns_none(self):
        orch = MultiAgentOrchestrator()
        result = orch.assign_task(Task(name='orphan'))
        self.assertIsNone(result)

    def test_planner_with_no_executor_records_failure(self):
        goal = Task(name='goal')
        planner = LongTaskPlanner(goal)  # 无 orchestrator，使用默认 worker
        # 默认 worker 存在，应正常执行
        planner.add_step(Task(name='step'), action=lambda t: 'ok')
        summary = planner.run()
        self.assertEqual(summary['status'], 'completed')

    def test_planner_with_orchestrator_no_workers(self):
        orch = MultiAgentOrchestrator()  # 无 worker
        goal = Task(name='goal')
        planner = LongTaskPlanner(goal, orchestrator=orch)
        planner.add_step(Task(name='step'), action=lambda t: 'ok')
        summary = planner.run()
        # 无 worker 时 assign_task 返回 None，planner 记录失败
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['failed_steps'], 1)


class TestReplanLoopGuard(unittest.TestCase):
    """循环 replan 边界：防止 replan 回调无限插入步骤"""

    def test_replan_respects_max_steps(self):
        call_count = {'n': 0}

        def infinite_replan(p, idx, record):
            call_count['n'] += 1
            return [Task(name=f'extra_{call_count["n"]}')]

        goal = Task(name='loop')
        planner = LongTaskPlanner(
            goal,
            max_steps=5,
            replan_callback=infinite_replan,
        )
        planner.add_step(Task(name='fail'), action=lambda t: 1 / 0, retries=0)
        summary = planner.run()
        # max_steps 限制执行次数，不会无限循环
        self.assertLessEqual(len(summary['results']), 5)
        self.assertIn(summary['status'], ('failed', 'paused', 'completed'))

    def test_replan_returning_empty_does_not_loop(self):
        goal = Task(name='no_replan')
        planner = LongTaskPlanner(
            goal,
            replan_callback=lambda p, idx, record: [],
        )
        planner.add_step(Task(name='fail'), action=lambda t: 1 / 0, retries=0)
        summary = planner.run()
        # replan 返回空列表，直接失败
        self.assertEqual(summary['status'], 'failed')


class TestCheckpointCorruption(unittest.TestCase):
    """checkpoint 损坏边界"""

    def test_resume_with_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(base_dir=os.path.join(tmp, 'ltm'))
            goal = Task(name='resume')
            planner = LongTaskPlanner(goal, memory=memory)
            # 未保存任何 checkpoint，resume 应返回 None
            self.assertIsNone(planner.resume())

    def test_resume_with_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = LongTermMemory(base_dir=os.path.join(tmp, 'ltm'))
            goal = Task(name='resume')
            planner = LongTaskPlanner(goal, memory=memory)
            # 写入损坏的 checkpoint（非 JSON 格式）
            knowledge_path = os.path.join(memory.knowledge_dir, f"{planner.plan_id}.json")
            with open(knowledge_path, 'w') as f:
                f.write("{corrupt json content")
            # resume 应优雅处理损坏数据
            try:
                planner.resume()
                # 若抛异常则测试失败；若返回 None 也算优雅处理
            except Exception as e:
                self.fail(f"resume should handle corrupt checkpoint gracefully, got: {e}")


class TestTaskWithoutCapabilities(unittest.TestCase):
    """任务无能力要求边界"""

    def test_task_without_required_capabilities_matches_any_worker(self):
        orch = MultiAgentOrchestrator(strategy='capability_match')
        orch.register_worker(AgentWorker(worker_id='w1', capabilities=['cpu']))
        task = Task(name='generic')  # 无 required_capabilities
        result = orch.assign_task(task)
        self.assertIsNotNone(result)
        self.assertEqual(result['worker_id'], 'w1')

    def test_skill_without_capabilities_handles_any_task(self):
        registry = SkillRegistry()
        registry.register(Skill(name='universal', action=lambda t: 'ok'))
        task = Task(name='any', params={'required_capabilities': ['cpu', 'disk', 'network']})
        # universal 技能无能力标签，can_handle 在无 required 时不拦截，
        # 但任务有 required_capabilities 时需要技能具备全部能力
        # 根据 Skill.can_handle 逻辑：required 非空时检查 all(req in capabilities)
        # universal capabilities 为空，故不能处理带 required 的任务
        self.assertIsNone(registry.select_best(task))

    def test_skill_without_capabilities_handles_generic_task(self):
        registry = SkillRegistry()
        registry.register(Skill(name='universal', action=lambda t: 'ok'))
        task = Task(name='generic')  # 无 required_capabilities
        best = registry.select_best(task)
        self.assertIsNotNone(best)
        self.assertEqual(best.name, 'universal')


class TestZeroStepPlan(unittest.TestCase):
    """零步骤计划边界"""

    def test_run_empty_plan(self):
        goal = Task(name='empty')
        planner = LongTaskPlanner(goal)
        summary = planner.run()
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['total_steps'], 0)
        self.assertEqual(summary['completed_steps'], 0)

    def test_runner_with_empty_steps(self):
        orch = MultiAgentOrchestrator()
        orch.register_worker(AgentWorker(worker_id='w1'))
        runner = MultiAgentLongTaskRunner(orch)
        summary = runner.run(Task(name='empty'), [])
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['total_steps'], 0)


class TestReviewerEdgeCases(unittest.TestCase):
    """复盘引擎边界"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = LongTermMemory(base_dir=os.path.join(self.tmp.name, 'ltm'))
        self.reviewer = LongTaskReviewer(self.memory)

    def tearDown(self):
        self.tmp.cleanup()

    def test_review_empty_results(self):
        summary = {'plan_id': 'empty', 'status': 'completed', 'results': []}
        report = self.reviewer.review(summary)
        self.assertEqual(report['total_steps'], 0)
        self.assertEqual(report['success_rate'], 0.0)
        self.assertEqual(len(report['failure_attribution']), 0)

    def test_review_load_nonexistent(self):
        self.assertIsNone(self.reviewer.load_review('nonexistent_id'))

    def test_review_history_empty(self):
        self.assertEqual(self.reviewer.review_history(), [])


if __name__ == '__main__':
    unittest.main()
