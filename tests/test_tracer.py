import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.tracer import CycleTracer, get_tracer


class TestCycleTracer(unittest.TestCase):
    def setUp(self):
        self.tracer = CycleTracer()
        self.tracer.clear()

    def test_trace(self):
        self.tracer.set_cycle(1)
        self.tracer.trace('TEST', 'hello')
        traces = self.tracer.get_recent_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]['phase'], 'TEST')
        self.assertEqual(traces[0]['message'], 'hello')

    def test_trace_with_data(self):
        self.tracer.set_cycle(2)
        self.tracer.trace('TEST', 'msg', {'key': 'value'})
        traces = self.tracer.get_recent_traces()
        self.assertEqual(traces[0]['data']['key'], 'value')

    def test_trace_block(self):
        self.tracer.set_cycle(1)
        with self.tracer.trace_block('TEST', 'block1'):
            pass

        traces = self.tracer.get_recent_traces()
        self.assertEqual(len(traces), 2)
        self.assertIn('START', traces[0]['message'])
        self.assertIn('END', traces[1]['message'])

    def test_trace_block_exception(self):
        self.tracer.set_cycle(1)
        try:
            with self.tracer.trace_block('TEST', 'bad_block'):
                raise ValueError('test error')
        except ValueError:
            pass

        traces = self.tracer.get_recent_traces()
        self.assertTrue(any('ERROR' in t['message'] for t in traces))

    def test_disable_enable(self):
        self.tracer.disable()
        self.tracer.trace('TEST', 'should not appear')
        self.assertEqual(len(self.tracer.get_recent_traces()), 0)

        self.tracer.enable()
        self.tracer.trace('TEST', 'should appear')
        self.assertEqual(len(self.tracer.get_recent_traces()), 1)

    def test_log_limit(self):
        self.tracer.set_cycle(1)
        for i in range(12000):
            self.tracer.trace('TEST', f'msg {i}')
        self.assertLessEqual(len(self.tracer._trace_logs), 5500)

    def test_get_by_phase(self):
        self.tracer.set_cycle(1)
        self.tracer.trace('A', 'msg1')
        self.tracer.trace('B', 'msg2')
        self.tracer.trace('A', 'msg3')

        a_traces = self.tracer.get_traces_by_phase('A')
        self.assertEqual(len(a_traces), 2)

    def test_stats(self):
        self.tracer.set_cycle(5)
        self.tracer.trace('A', 'msg1')
        self.tracer.trace('B', 'msg2')
        stats = self.tracer.get_stats()
        self.assertEqual(stats['current_cycle'], 5)
        self.assertEqual(stats['total_traces'], 2)





if __name__ == '__main__':
    unittest.main()
