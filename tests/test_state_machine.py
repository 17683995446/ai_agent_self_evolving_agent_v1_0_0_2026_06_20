import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.state_machine import StateMachine, AgentState, StateTransition


class TestStateMachine(unittest.TestCase):
    def test_initial_state(self):
        sm = StateMachine()
        self.assertEqual(sm.state, AgentState.IDLE)
        self.assertTrue(sm.is_operational())
        self.assertFalse(sm.is_critical())

    def test_valid_transition(self):
        sm = StateMachine()
        result = sm.transition(AgentState.ACTIVE, "start")
        self.assertTrue(result)
        self.assertEqual(sm.state, AgentState.ACTIVE)
        self.assertEqual(sm.previous_state, AgentState.IDLE)

    def test_invalid_transition(self):
        sm = StateMachine(AgentState.SHUTDOWN)
        result = sm.transition(AgentState.ACTIVE, "invalid")
        self.assertFalse(result)
        self.assertEqual(sm.state, AgentState.SHUTDOWN)

    def test_can_transition(self):
        sm = StateMachine(AgentState.ACTIVE)
        self.assertTrue(sm.can_transition(AgentState.EMERGENCY))
        self.assertTrue(sm.can_transition(AgentState.REVIEW))
        self.assertTrue(sm.can_transition(AgentState.IDLE))
        self.assertTrue(sm.can_transition(AgentState.SHUTDOWN))

    def test_history(self):
        sm = StateMachine()
        sm.transition(AgentState.ACTIVE, "start")
        sm.transition(AgentState.REVIEW, "review")
        history = sm.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].from_state, AgentState.IDLE)
        self.assertEqual(history[0].to_state, AgentState.ACTIVE)

    def test_state_duration(self):
        sm = StateMachine()
        import time
        time.sleep(0.05)
        self.assertGreaterEqual(sm.get_state_duration(), 0.05)

    def test_handler_called(self):
        sm = StateMachine()
        called = []

        def handler(from_state, to_state):
            called.append((from_state, to_state))

        sm.on_enter(AgentState.ACTIVE, handler)
        sm.transition(AgentState.ACTIVE, "test")

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][1], AgentState.ACTIVE)

    def test_transition_handler_called(self):
        sm = StateMachine()
        transitions = []

        def handler(transition):
            transitions.append(transition)

        sm.on_transition(handler)
        sm.transition(AgentState.ACTIVE, "test")

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].reason, "test")

    def test_auto_transition_emergency(self):
        sm = StateMachine(AgentState.ACTIVE)
        result = sm.auto_transition({'emergency_mode': True})
        self.assertIsNotNone(result)
        self.assertEqual(sm.state, AgentState.EMERGENCY)

    def test_auto_transition_recovery(self):
        sm = StateMachine(AgentState.EMERGENCY)
        result = sm.auto_transition({'emergency_mode': False})
        self.assertIsNotNone(result)
        self.assertEqual(sm.state, AgentState.ACTIVE)

    def test_auto_transition_review(self):
        sm = StateMachine(AgentState.ACTIVE)
        result = sm.auto_transition({'emergency_mode': False}, review_ready=True)
        self.assertIsNotNone(result)
        self.assertEqual(sm.state, AgentState.REVIEW)

    def test_auto_transition_maintenance(self):
        sm = StateMachine(AgentState.ACTIVE)
        result = sm.auto_transition({'emergency_mode': False}, maintenance_requested=True)
        self.assertIsNotNone(result)
        self.assertEqual(sm.state, AgentState.MAINTENANCE)

    def test_auto_transition_idle_to_active(self):
        sm = StateMachine(AgentState.IDLE)
        result = sm.auto_transition({'emergency_mode': False})
        self.assertIsNotNone(result)
        self.assertEqual(sm.state, AgentState.ACTIVE)

    def test_no_duplicate_emergency(self):
        sm = StateMachine(AgentState.EMERGENCY)
        result = sm.auto_transition({'emergency_mode': True})
        self.assertIsNone(result)

    def test_stats(self):
        sm = StateMachine()
        sm.transition(AgentState.ACTIVE, "start")
        stats = sm.get_stats()
        self.assertEqual(stats['current_state'], 'ACTIVE')
        self.assertEqual(stats['previous_state'], 'IDLE')
        self.assertEqual(stats['total_transitions'], 1)
        self.assertTrue(stats['is_operational'])
        self.assertFalse(stats['is_critical'])


class TestAgentState(unittest.TestCase):
    def test_state_values(self):
        self.assertEqual(AgentState.IDLE.name, 'IDLE')
        self.assertEqual(AgentState.ACTIVE.name, 'ACTIVE')
        self.assertEqual(AgentState.EMERGENCY.name, 'EMERGENCY')


class TestStateTransition(unittest.TestCase):
    def test_transition_creation(self):
        t = StateTransition(
            from_state=AgentState.IDLE,
            to_state=AgentState.ACTIVE,
            reason="test"
        )
        self.assertEqual(t.from_state, AgentState.IDLE)
        self.assertEqual(t.to_state, AgentState.ACTIVE)
        self.assertEqual(t.reason, "test")
        self.assertGreater(t.timestamp, 0)


if __name__ == '__main__':
    unittest.main()
