import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.task.task_scorer import TaskScorer
from src.task.task_definition import Task
from src.core.constants import TaskType


class TestTaskScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = TaskScorer()
        self.state = {}
        self.gap_vector = {'safe_gap': 0.5, 'gain_gap': 0.3, 'eff_gap': 0.2}

    def test_score_frozen_task(self):
        task = Task(type=TaskType.SAFE)
        task.freeze()
        score = self.scorer.score_task(task, self.state, self.gap_vector)
        self.assertEqual(score, -1.0)

    def test_score_unfrozen_task(self):
        task = Task(type=TaskType.SAFE)
        score = self.scorer.score_task(task, self.state, self.gap_vector)
        self.assertGreater(score, 0.0)

    def test_type_match_affects_score(self):
        safe_task = Task(type=TaskType.SAFE)
        gain_task = Task(type=TaskType.GAIN)
        safe_score = self.scorer.score_task(safe_task, self.state, self.gap_vector)
        gain_score = self.scorer.score_task(gain_task, self.state, self.gap_vector)
        self.assertGreater(safe_score, gain_score)

    def test_success_rate_affects_score(self):
        task1 = Task(type=TaskType.SAFE)
        task1.mark_executed(success=True)
        task1.mark_executed(success=True)

        task2 = Task(type=TaskType.SAFE)
        task2.mark_executed(success=False)
        task2.mark_executed(success=False)

        score1 = self.scorer.score_task(task1, self.state, self.gap_vector)
        score2 = self.scorer.score_task(task2, self.state, self.gap_vector)
        self.assertGreater(score1, score2)

    def test_freshness_bonus(self):
        new_task = Task(type=TaskType.SAFE)
        old_task = Task(type=TaskType.SAFE)
        old_task.mark_executed(success=False)
        old_task.mark_executed(success=False)

        new_score = self.scorer.score_task(new_task, self.state, self.gap_vector)
        old_score = self.scorer.score_task(old_task, self.state, self.gap_vector)
        self.assertGreater(new_score, old_score)

    def test_time_decay(self):
        task = Task(type=TaskType.SAFE)
        task.mark_executed(success=True)
        task.last_executed = time.time() - 7200  # 2 hours ago

        score = self.scorer.score_task(task, self.state, self.gap_vector)
        self.assertLess(score, 1.0)

    def test_score_all(self):
        tasks = [
            Task(type=TaskType.SAFE),
            Task(type=TaskType.GAIN),
            Task(type=TaskType.EFF),
        ]
        scores = self.scorer.score_all(tasks, self.state, self.gap_vector)
        self.assertEqual(len(scores), 3)
        for task in tasks:
            self.assertIn(task.id, scores)
            self.assertEqual(task.score, scores[task.id])

    def test_rank_tasks(self):
        tasks = [
            Task(type=TaskType.SAFE, name='low'),
            Task(type=TaskType.GAIN, name='high'),
            Task(type=TaskType.EFF, name='medium'),
        ]
        self.scorer.score_all(tasks, self.state, self.gap_vector)
        ranked = self.scorer.rank_tasks(tasks)
        self.assertEqual(ranked[0].type, TaskType.SAFE)
        self.assertEqual(ranked[-1].type, TaskType.EFF)

    def test_score_updates_task(self):
        task = Task(type=TaskType.SAFE)
        self.assertEqual(task.score, 0.0)
        self.scorer.score_task(task, self.state, self.gap_vector)
        self.assertNotEqual(task.score, 0.0)


if __name__ == '__main__':
    unittest.main()
