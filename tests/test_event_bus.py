import os
import sys
import unittest
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.event_bus import Event, EventBus, AsyncTaskScheduler


class TestEvent(unittest.TestCase):
    def test_event_creation(self):
        e = Event(type='test', payload={'key': 'value'}, source='unit_test')
        self.assertEqual(e.type, 'test')
        self.assertEqual(e.payload['key'], 'value')
        self.assertEqual(e.source, 'unit_test')
        self.assertGreater(e.timestamp, 0)

    def test_event_to_dict(self):
        e = Event(type='test', payload={'a': 1})
        d = e.to_dict()
        self.assertEqual(d['type'], 'test')
        self.assertEqual(d['payload']['a'], 1)


class TestEventBus(unittest.IsolatedAsyncioTestCase):
    async def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe('test_event', handler)
        await bus.start()

        event = Event(type='test_event', payload={'msg': 'hello'})
        await bus.publish(event)

        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload['msg'], 'hello')

    async def test_unsubscribe(self):
        bus = EventBus()

        async def handler(event):
            pass

        bus.subscribe('test', handler)
        self.assertTrue(bus.unsubscribe('test', handler))
        self.assertFalse(bus.unsubscribe('test', handler))

    async def test_multiple_handlers(self):
        bus = EventBus()
        counts = [0, 0]

        async def h1(event):
            counts[0] += 1

        async def h2(event):
            counts[1] += 1

        bus.subscribe('multi', h1)
        bus.subscribe('multi', h2)
        await bus.start()

        await bus.publish(Event(type='multi'))
        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[1], 1)

    async def test_wildcard_subscription(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event.type)

        bus.subscribe('*', handler)
        await bus.start()

        await bus.publish(Event(type='event_a'))
        await bus.publish(Event(type='event_b'))
        await asyncio.sleep(0.1)
        await bus.stop()

        self.assertIn('event_a', received)
        self.assertIn('event_b', received)

    async def test_no_handler_no_crash(self):
        bus = EventBus()
        await bus.start()
        await bus.publish(Event(type='unknown'))
        await asyncio.sleep(0.1)
        await bus.stop()

    async def test_stats(self):
        bus = EventBus()
        async def handler(event):
            pass
        bus.subscribe('t1', handler)
        bus.subscribe('t2', handler)
        stats = bus.get_stats()
        self.assertEqual(stats['total_handlers'], 2)


class TestAsyncTaskScheduler(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_task(self):
        scheduler = AsyncTaskScheduler()
        await scheduler.start()

        async def my_task():
            return 42

        task = await scheduler.schedule(my_task(), task_id='test_1')
        result = await task
        await scheduler.stop()

        self.assertEqual(result, 42)

    async def test_max_concurrent(self):
        scheduler = AsyncTaskScheduler(max_concurrent=2)
        await scheduler.start()

        running = []

        async def slow_task():
            running.append(1)
            await asyncio.sleep(0.2)
            running.pop()
            return len(running)

        tasks = []
        for i in range(5):
            t = await scheduler.schedule(slow_task())
            tasks.append(t)

        await asyncio.sleep(0.05)
        # 由于信号量限制，同时运行的不应超过2个
        self.assertLessEqual(len(running), 2)

        await asyncio.gather(*tasks)
        await scheduler.stop()

    async def test_error_handling(self):
        scheduler = AsyncTaskScheduler()
        await scheduler.start()

        async def bad_task():
            raise ValueError("intentional error")

        task = await scheduler.schedule(bad_task())
        with self.assertRaises(ValueError):
            await task

        await scheduler.stop()

    async def test_stop_cancels_tasks(self):
        scheduler = AsyncTaskScheduler()
        await scheduler.start()

        async def long_task():
            await asyncio.sleep(10)
            return 1

        await scheduler.schedule(long_task())
        await scheduler.stop()

        stats = scheduler.get_stats()
        self.assertEqual(stats['active_tasks'], 0)


if __name__ == '__main__':
    unittest.main()
