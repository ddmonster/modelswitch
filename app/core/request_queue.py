"""请求队列管理器 - 支持 provider 级别的并发控制"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """队列中的请求"""

    id: str
    func: Callable[..., Coroutine[Any, Any, Any]]
    args: tuple
    kwargs: Dict[str, Any]
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.time)
    priority: int = 0  # 优先级，数字越小越优先


class ProviderQueue:
    """单个 Provider 的请求队列和并发控制器"""

    def __init__(
        self,
        provider_name: str,
        max_concurrent: int = 1,
        max_queue_size: int = 100,
        queue_timeout: float = 300.0,
    ):
        self.provider_name = provider_name
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.queue_timeout = queue_timeout

        # 并发控制
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._request_map: Dict[str, QueuedRequest] = {}
        self._counter = 0

        # 统计
        self._stats = {
            "total_requests": 0,
            "queued_requests": 0,
            "rejected_requests": 0,
            "avg_wait_time": 0.0,
        }

        # 启动队列处理器
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def update_config(
        self,
        max_concurrent: int,
        max_queue_size: int,
        queue_timeout: float,
    ):
        """更新队列配置，重建信号量以反映新的并发限制。"""
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.queue_timeout = queue_timeout
        # 用新限制重建信号量。当前限制约为允许的最大值，
        # 因此新信号量从 max_concurrent 减去当前正在进行的请求数开始。
        current_active = self.max_concurrent - self._semaphore._value
        self._semaphore = asyncio.Semaphore(max(0, max_concurrent - current_active))
        logger.info(
            f"[{self.provider_name}] 队列配置已更新: 并发={max_concurrent}, 队列={max_queue_size}, 超时={queue_timeout}s"
        )

    async def start(self):
        """启动队列处理器"""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._process_queue())
            logger.info(
                f"[{self.provider_name}] 请求队列已启动，最大并发: {self.max_concurrent}"
            )

    async def stop(self):
        """停止队列处理器"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # 取消队列中所有等待的请求
        while not self._queue.empty():
            try:
                _, _, request = self._queue.get_nowait()
                if not request.future.done():
                    request.future.set_exception(
                        RuntimeError("Provider queue is shutting down")
                    )
            except asyncio.QueueEmpty:
                break

        logger.info(f"[{self.provider_name}] 请求队列已停止")

    async def enqueue(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        priority: int = 0,
        **kwargs,
    ) -> asyncio.Future:
        """将请求加入队列"""
        self._counter += 1
        request_id = f"{self.provider_name}_{self._counter}_{time.time()}"

        # 检查队列是否已满
        if self._queue.qsize() >= self.max_queue_size:
            self._stats["rejected_requests"] += 1
            raise RuntimeError(
                f"[{self.provider_name}] 请求队列已满 ({self.max_queue_size}), "
                f"请稍后重试"
            )

        future = asyncio.get_event_loop().create_future()
        request = QueuedRequest(
            id=request_id,
            func=func,
            args=args,
            kwargs=kwargs,
            future=future,
            priority=priority,
        )

        # 使用优先级队列 (priority, counter, request)
        await self._queue.put((priority, self._counter, request))
        self._request_map[request_id] = request
        self._stats["total_requests"] += 1
        self._stats["queued_requests"] = self._queue.qsize()

        wait_time = self._estimate_wait_time()
        logger.debug(
            f"[{self.provider_name}] 请求已入队 #{self._counter}, "
            f"当前队列长度: {self._queue.qsize()}, 预估等待: {wait_time:.1f}s"
        )

        return future

    def _estimate_wait_time(self) -> float:
        """估算等待时间"""
        queue_size = self._queue.qsize()
        # 简单估算：每个请求平均 5 秒
        return queue_size * 5.0 / max(1, self.max_concurrent)

    async def _process_queue(self):
        """队列处理器主循环"""
        while self._running:
            try:
                # 获取队列中的请求
                _, _, request = await self._queue.get()

                # 检查是否超时
                wait_time = time.time() - request.enqueue_time
                if wait_time > self.queue_timeout:
                    if not request.future.done():
                        request.future.set_exception(
                            asyncio.TimeoutError(
                                f"请求在队列中等待超时 ({wait_time:.1f}s > {self.queue_timeout}s)"
                            )
                        )
                    self._queue.task_done()
                    continue

                # 使用信号量控制并发
                asyncio.create_task(self._execute_with_semaphore(request))
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.provider_name}] 队列处理错误: {e}")

    async def _execute_with_semaphore(self, request: QueuedRequest):
        """在信号量控制下执行请求"""
        async with self._semaphore:
            try:
                # 检查 future 是否已被取消
                if request.future.done():
                    return

                wait_time = time.time() - request.enqueue_time
                logger.debug(
                    f"[{self.provider_name}] 开始执行请求 #{request.id}, 等待时间: {wait_time:.2f}s"
                )

                # 执行实际请求
                result = await request.func(*request.args, **request.kwargs)

                if not request.future.done():
                    request.future.set_result(result)

                # 更新统计
                self._update_avg_wait_time(wait_time)

            except Exception as e:
                if not request.future.done():
                    request.future.set_exception(e)
                logger.error(f"[{self.provider_name}] 请求执行错误: {e}")

    def _update_avg_wait_time(self, wait_time: float):
        """更新平均等待时间"""
        alpha = 0.1  # 指数移动平均系数
        self._stats["avg_wait_time"] = (
            alpha * wait_time + (1 - alpha) * self._stats["avg_wait_time"]
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        return {
            "provider": self.provider_name,
            "max_concurrent": self.max_concurrent,
            "current_queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_requests": self.max_concurrent - self._semaphore._value,
            "queue_timeout": self.queue_timeout,
            **self._stats,
        }


class RequestQueueManager:
    """全局请求队列管理器"""

    def __init__(self):
        self._queues: Dict[str, ProviderQueue] = {}
        self._default_max_concurrent = 1
        self._default_max_queue_size = 100

    def register_provider(
        self,
        provider_name: str,
        max_concurrent: int = 1,
        max_queue_size: int = 100,
        queue_timeout: float = 300.0,
    ) -> ProviderQueue:
        """注册或更新 provider 队列"""
        if provider_name in self._queues:
            queue = self._queues[provider_name]
            queue.update_config(max_concurrent, max_queue_size, queue_timeout)
            return queue

        queue = ProviderQueue(
            provider_name=provider_name,
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
            queue_timeout=queue_timeout,
        )
        self._queues[provider_name] = queue
        return queue

    async def unregister_provider(self, provider_name: str):
        """注销 provider 队列（当 max_concurrent 设为 0 时调用）"""
        queue = self._queues.pop(provider_name, None)
        if queue:
            await queue.stop()
            logger.info(f"[{provider_name}] 请求队列已注销")

    def sync_providers(
        self,
        providers,  # list of provider configs with .name, .max_concurrent, .max_queue_size, .queue_timeout
    ):
        """同步队列管理器以匹配提供者配置。

        - 注册或更新 max_concurrent > 0 的提供者
        - 取消注册 max_concurrent == 0 的提供者
        返回需要启动的队列列表。
        """
        registered = set()
        for p in providers:
            registered.add(p.name)
            self.register_provider(
                provider_name=p.name,
                max_concurrent=p.max_concurrent,
                max_queue_size=p.max_queue_size,
                queue_timeout=p.queue_timeout,
            )

        # 注销已移除或已禁用的提供者（max_concurrent == 0）
        to_remove = [name for name in self._queues if name not in registered]
        return to_remove

    async def start(self):
        """启动所有队列"""
        for queue in self._queues.values():
            await queue.start()

    async def stop(self):
        """停止所有队列"""
        for queue in self._queues.values():
            await queue.stop()

    async def execute(
        self,
        provider_name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        priority: int = 0,
        **kwargs,
    ) -> Any:
        """在指定 provider 的队列中执行请求"""
        queue = self._queues.get(provider_name)
        if not queue:
            # 如果没有配置队列，直接执行
            return await func(*args, **kwargs)

        future = await queue.enqueue(func, *args, priority=priority, **kwargs)
        return await future

    def get_queue(self, provider_name: str) -> Optional[ProviderQueue]:
        """获取指定 provider 的队列"""
        return self._queues.get(provider_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有队列的统计信息"""
        return {name: queue.get_stats() for name, queue in self._queues.items()}


# 全局队列管理器实例
_queue_manager: Optional[RequestQueueManager] = None


def get_queue_manager() -> RequestQueueManager:
    """获取全局队列管理器实例"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = RequestQueueManager()
    return _queue_manager


def reset_queue_manager():
    """重置队列管理器（用于测试）"""
    global _queue_manager
    _queue_manager = None
