import abc
import asyncio
import logging
from typing import Dict, Any, List, Callable, Awaitable, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """标准化事件"""
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    source: str = "unknown"
    priority: int = 5  # 1-10, 1最高

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。

        :param: 无
        :type: 无
        :return: 事件字典。
        :rtype: Dict[str, Any]
        """
        return {
            'type': self.type,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'source': self.source,
            'priority': self.priority,
        }


class BaseEventBus(abc.ABC):
    """异步事件总线基类"""

    def __init__(self) -> None:
        """初始化事件总线。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._running = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    @abc.abstractmethod
    def subscribe(self, event_type: str,
                  handler: Callable[[Event], Awaitable[None]]) -> None:
        """订阅事件。

        :param event_type: 事件类型。
        :type event_type: str
        :param handler: 异步处理函数。
        :type handler: Callable[[Event], Awaitable[None]]
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    def unsubscribe(self, event_type: str,
                    handler: Callable[[Event], Awaitable[None]]) -> bool:
        """取消订阅。

        :param event_type: 事件类型。
        :type event_type: str
        :param handler: 要移除的处理函数。
        :type handler: Callable[[Event], Awaitable[None]]
        :return: 是否成功移除。
        :rtype: bool
        """

    @abc.abstractmethod
    async def publish(self, event: Event) -> None:
        """发布事件到队列。

        :param event: 要发布的事件。
        :type event: Event
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    def publish_sync(self, event: Event) -> None:
        """同步发布（用于非异步上下文）。

        :param event: 要发布的事件。
        :type event: Event
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    async def start(self) -> None:
        """启动事件分发循环。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """停止事件分发。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """


class EventBus(BaseEventBus):
    """异步事件总线：解耦模块间通信"""

    def subscribe(self, event_type: str,
                  handler: Callable[[Event], Awaitable[None]]) -> None:
        """订阅事件。

        :param event_type: 事件类型。
        :type event_type: str
        :param handler: 异步处理函数。
        :type handler: Callable[[Event], Awaitable[None]]
        :return: 无。
        :rtype: None
        """
        self._subscribers[event_type].append(handler)
        logger.debug("Handler subscribed to %s", event_type)

    def unsubscribe(self, event_type: str,
                    handler: Callable[[Event], Awaitable[None]]) -> bool:
        """取消订阅。

        :param event_type: 事件类型。
        :type event_type: str
        :param handler: 要移除的处理函数。
        :type handler: Callable[[Event], Awaitable[None]]
        :return: 是否成功移除。
        :rtype: bool
        """
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            return True
        return False

    async def publish(self, event: Event) -> None:
        """发布事件到队列。

        :param event: 要发布的事件。
        :type event: Event
        :return: 无。
        :rtype: None
        """
        await self._queue.put(event)
        logger.debug("Event %s published", event.type)

    def publish_sync(self, event: Event) -> None:
        """同步发布（用于非异步上下文）。

        :param event: 要发布的事件。
        :type event: Event
        :return: 无。
        :rtype: None
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._queue.put(event))
        except RuntimeError as e:
            logger.debug("EventBus.publish_async: no running loop, event dropped: %s", e)

    async def start(self) -> None:
        """启动事件分发循环。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())
        logger.info("EventBus started")

    async def stop(self) -> None:
        """停止事件分发。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("EventBus.stop: dispatch task cancelled")
        logger.info("EventBus stopped")

    async def _dispatch_loop(self) -> None:
        """事件分发主循环。"""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Event dispatch error: %s", e)

    async def _dispatch(self, event: Event) -> None:
        """分发单个事件到所有订阅者。

        :param event: 要分发的事件。
        :type event: Event
        """
        handlers = self._get_handlers(event.type)
        for handler in handlers:
            await self._safe_handle(handler, event)

    def _get_handlers(self, event_type: str) -> List[Callable[[Event], Awaitable[None]]]:
        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            handlers = self._subscribers.get('*', [])
        return handlers

    async def _safe_handle(self, handler: Callable[[Event], Awaitable[None]], event: Event) -> None:
        try:
            await handler(event)
        except Exception as e:
            logger.error("Handler error for %s: %s", event.type, e)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """
        return {
            'subscribed_types': list(self._subscribers.keys()),
            'total_handlers': sum(len(h) for h in self._subscribers.values()),
            'queue_size': self._queue.qsize(),
            'running': self._running,
        }


class BaseAsyncTaskScheduler(abc.ABC):
    """异步任务调度器基类"""

    def __init__(self, max_concurrent: int = 5) -> None:
        """初始化异步任务调度器。

        :param max_concurrent: 最大并发数。
        :type max_concurrent: int
        :return: 无。
        :rtype: None
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set = set()
        self._running = False

    @abc.abstractmethod
    async def schedule(self, coro: Awaitable[Any],
                       task_id: Optional[str] = None) -> asyncio.Task:
        """调度一个异步任务。

        :param coro: 协程对象。
        :type coro: Awaitable[Any]
        :param task_id: 任务标识。
        :type task_id: Optional[str]
        :return: 创建的任务对象。
        :rtype: asyncio.Task
        """

    @abc.abstractmethod
    async def start(self) -> None:
        """启动调度器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """停止调度器并取消所有任务。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """


class AsyncTaskScheduler(BaseAsyncTaskScheduler):
    """异步任务调度器：管理后台任务"""

    async def schedule(self, coro: Awaitable[Any],
                       task_id: Optional[str] = None) -> asyncio.Task:
        """调度一个异步任务。

        :param coro: 协程对象。
        :type coro: Awaitable[Any]
        :param task_id: 任务标识。
        :type task_id: Optional[str]
        :return: 创建的任务对象。
        :rtype: asyncio.Task
        """
        task = asyncio.create_task(self._wrapped_coro(coro))
        self._register_task(task, task_id)
        return task

    async def _wrapped_coro(self, coro: Awaitable[Any]) -> Any:
        async with self._semaphore:
            try:
                return await coro
            except Exception as e:
                logger.error("Scheduled task error: %s", e)
                raise

    def _register_task(self, task: asyncio.Task, task_id: Optional[str]) -> None:
        task._agent_task_id = task_id or f"task_{id(task)}"
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def start(self) -> None:
        """启动调度器。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._running = True
        logger.info("TaskScheduler started")

    async def stop(self) -> None:
        """停止调度器并取消所有任务。

        :param: 无
        :type: 无
        :return: 无。
        :rtype: None
        """
        self._running = False
        if self._tasks:
            for task in list(self._tasks):
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        logger.info("TaskScheduler stopped")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        :param: 无
        :type: 无
        :return: 统计字典。
        :rtype: Dict[str, Any]
        """
        return {
            'active_tasks': len(self._tasks),
            'max_concurrent': self.max_concurrent,
            'running': self._running,
        }
