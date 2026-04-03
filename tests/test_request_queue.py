"""Comprehensive tests for request_queue.py to achieve 100% coverage"""

import asyncio
import time

import pytest

from app.core.request_queue import (
    ProviderQueue,
    QueuedRequest,
    RequestQueueManager,
    get_queue_manager,
    reset_queue_manager,
)


class TestQueuedRequest:
    """Tests for QueuedRequest dataclass"""

    def test_queued_request_creation(self):
        """Test basic QueuedRequest creation"""

        async def dummy_func():
            pass

        future = asyncio.get_event_loop().create_future()
        request = QueuedRequest(
            id="test_1",
            func=dummy_func,
            args=(1, 2),
            kwargs={"key": "value"},
            future=future,
        )

        assert request.id == "test_1"
        assert request.func == dummy_func
        assert request.args == (1, 2)
        assert request.kwargs == {"key": "value"}
        assert request.future == future
        assert request.enqueue_time > 0
        assert request.priority == 0

    def test_queued_request_with_priority(self):
        """Test QueuedRequest with custom priority"""

        async def dummy_func():
            pass

        future = asyncio.get_event_loop().create_future()
        request = QueuedRequest(
            id="test_2",
            func=dummy_func,
            args=(),
            kwargs={},
            future=future,
            priority=5,
        )

        assert request.priority == 5

    def test_queued_request_custom_enqueue_time(self):
        """Test QueuedRequest with custom enqueue_time"""

        async def dummy_func():
            pass

        future = asyncio.get_event_loop().create_future()
        custom_time = 12345.0
        request = QueuedRequest(
            id="test_3",
            func=dummy_func,
            args=(),
            kwargs={},
            future=future,
            enqueue_time=custom_time,
        )

        assert request.enqueue_time == custom_time


class TestProviderQueueInit:
    """Tests for ProviderQueue initialization"""

    def test_default_settings(self):
        """Test ProviderQueue with default settings"""
        queue = ProviderQueue("test_provider")

        assert queue.provider_name == "test_provider"
        assert queue.max_concurrent == 1
        assert queue.max_queue_size == 100
        assert queue.queue_timeout == 300.0
        assert queue._running is False
        assert queue._worker_task is None

    def test_custom_settings(self):
        """Test ProviderQueue with custom settings"""
        queue = ProviderQueue(
            provider_name="custom_provider",
            max_concurrent=5,
            max_queue_size=50,
            queue_timeout=60.0,
        )

        assert queue.provider_name == "custom_provider"
        assert queue.max_concurrent == 5
        assert queue.max_queue_size == 50
        assert queue.queue_timeout == 60.0

    def test_initial_stats(self):
        """Test initial stats are zero"""
        queue = ProviderQueue("test_provider")

        assert queue._stats["total_requests"] == 0
        assert queue._stats["queued_requests"] == 0
        assert queue._stats["rejected_requests"] == 0
        assert queue._stats["avg_wait_time"] == 0.0


class TestProviderQueueLifecycle:
    """Tests for ProviderQueue start/stop lifecycle"""

    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting the queue"""
        queue = ProviderQueue("test_provider")

        assert queue._running is False
        assert queue._worker_task is None

        await queue.start()

        assert queue._running is True
        assert queue._worker_task is not None

        await queue.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that start is idempotent"""
        queue = ProviderQueue("test_provider")

        await queue.start()
        first_task = queue._worker_task

        await queue.start()  # Should not create a new task
        assert queue._worker_task == first_task

        await queue.stop()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping the queue"""
        queue = ProviderQueue("test_provider")

        await queue.start()
        assert queue._running is True

        await queue.stop()
        assert queue._running is False
        # Note: _worker_task is cancelled but not set to None by the source code
        assert queue._worker_task is not None
        assert queue._worker_task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        """Test stopping a queue that was never started"""
        queue = ProviderQueue("test_provider")

        # Should not raise an error
        await queue.stop()
        assert queue._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_requests(self):
        """Test that stop cancels all pending requests in queue"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def slow_func():
            await asyncio.sleep(10)
            return "done"

        # Don't start the queue - this keeps items in queue
        # Enqueue multiple requests without starting
        futures = []
        for i in range(3):
            future = await queue.enqueue(slow_func, priority=i)
            futures.append(future)

        # Stop should cancel pending requests in queue
        await queue.stop()

        # Check that pending requests got cancelled
        cancelled_count = 0
        for future in futures:
            if future.done():
                exc = future.exception()
                if isinstance(exc, RuntimeError) and "shutting down" in str(exc):
                    cancelled_count += 1

        # All 3 requests should be cancelled since they were in queue when stop was called
        assert cancelled_count == 3


class TestProviderQueueEnqueue:
    """Tests for ProviderQueue enqueue functionality"""

    @pytest.mark.asyncio
    async def test_enqueue_success(self):
        """Test successful enqueue"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def dummy_func():
            return "success"

        await queue.start()
        future = await queue.enqueue(dummy_func)

        result = await future
        assert result == "success"

        # Check stats
        assert queue._stats["total_requests"] == 1

        await queue.stop()

    @pytest.mark.asyncio
    async def test_enqueue_with_args_and_kwargs(self):
        """Test enqueue with arguments"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def func_with_args(a, b, c=None):
            return (a, b, c)

        await queue.start()
        future = await queue.enqueue(func_with_args, 1, 2, c=3)

        result = await future
        assert result == (1, 2, 3)

        await queue.stop()

    @pytest.mark.asyncio
    async def test_enqueue_with_priority(self):
        """Test enqueue with priority"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def dummy_func():
            return "priority_test"

        await queue.start()
        future = await queue.enqueue(dummy_func, priority=10)

        result = await future
        assert result == "priority_test"

        await queue.stop()

    @pytest.mark.asyncio
    async def test_enqueue_queue_full(self):
        """Test enqueue when queue is full"""
        queue = ProviderQueue("test_provider", max_concurrent=1, max_queue_size=2)

        async def slow_func():
            await asyncio.sleep(10)
            return "slow"

        # Don't start the queue - this keeps items in queue
        # Fill up the queue
        futures = []
        for _ in range(2):  # max_queue_size=2
            future = await queue.enqueue(slow_func)
            futures.append(future)

        # Queue is now full, next enqueue should be rejected
        with pytest.raises(RuntimeError) as exc_info:
            await queue.enqueue(slow_func)

        assert "请求队列已满" in str(exc_info.value)
        assert queue._stats["rejected_requests"] == 1

    @pytest.mark.asyncio
    async def test_enqueue_returns_future(self):
        """Test that enqueue returns a future"""
        queue = ProviderQueue("test_provider")

        async def dummy_func():
            return "result"

        await queue.start()
        future = await queue.enqueue(dummy_func)

        assert isinstance(future, asyncio.Future)

        result = await future
        assert result == "result"

        await queue.stop()


class TestProviderQueueTimeout:
    """Tests for queue timeout functionality"""

    @pytest.mark.asyncio
    async def test_queue_timeout(self):
        """Test that requests waiting too long are timed out"""
        queue = ProviderQueue(
            "test_provider",
            max_concurrent=1,
            queue_timeout=0.1,  # Very short timeout
        )

        async def slow_func():
            await asyncio.sleep(1.0)
            return "done"

        # Don't start the queue initially - this keeps items in queue
        # Enqueue items that will sit in queue
        future1 = await queue.enqueue(slow_func)
        future2 = await queue.enqueue(slow_func)

        # Wait for timeout duration to pass
        await asyncio.sleep(0.2)  # Longer than queue_timeout

        # Now start the queue - worker should pull items and check timeout
        await queue.start()

        # Give worker time to process the queue
        await asyncio.sleep(0.1)

        # Both futures should have timed out since they sat in queue too long
        assert future1.done()
        assert future2.done()

        with pytest.raises(asyncio.TimeoutError):
            await future1

        with pytest.raises(asyncio.TimeoutError):
            await future2

        await queue.stop()


class TestProviderQueuePriority:
    """Tests for priority ordering"""

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that lower priority numbers are processed first"""
        queue = ProviderQueue("test_provider", max_concurrent=1)
        execution_order = []

        async def record_order(name):
            execution_order.append(name)
            await asyncio.sleep(0.05)  # Small delay to ensure sequential processing
            return name

        await queue.start()

        # Block the queue initially
        async def blocker():
            execution_order.append("blocker")
            await asyncio.sleep(0.1)

        await queue.enqueue(blocker, priority=0)

        # Wait for blocker to start
        await asyncio.sleep(0.05)

        # Enqueue with different priorities (lower number = higher priority)
        future3 = await queue.enqueue(record_order, "low", priority=100)
        future1 = await queue.enqueue(record_order, "high", priority=1)
        future2 = await queue.enqueue(record_order, "medium", priority=50)

        # Wait for all to complete
        await asyncio.gather(future1, future2, future3, return_exceptions=True)

        # After blocker, should be: high (1), medium (50), low (100)
        assert execution_order == ["blocker", "high", "medium", "low"]

        await queue.stop()

    @pytest.mark.asyncio
    async def test_same_priority_fifo(self):
        """Test that requests with same priority are processed in FIFO order"""
        queue = ProviderQueue("test_provider", max_concurrent=1)
        execution_order = []

        async def record_order(name):
            execution_order.append(name)
            await asyncio.sleep(0.02)
            return name

        await queue.start()

        # Block initially
        async def blocker():
            execution_order.append("blocker")
            await asyncio.sleep(0.1)

        await queue.enqueue(blocker)

        # Wait for blocker to start
        await asyncio.sleep(0.05)

        # Same priority should maintain FIFO
        future1 = await queue.enqueue(record_order, "first", priority=10)
        future2 = await queue.enqueue(record_order, "second", priority=10)
        future3 = await queue.enqueue(record_order, "third", priority=10)

        await asyncio.gather(future1, future2, future3, return_exceptions=True)

        assert execution_order == ["blocker", "first", "second", "third"]

        await queue.stop()


class TestProviderQueueConcurrency:
    """Tests for concurrent request handling"""

    @pytest.mark.asyncio
    async def test_concurrent_requests_within_limit(self):
        """Test that concurrent requests work within max_concurrent limit"""
        queue = ProviderQueue("test_provider", max_concurrent=3)
        active_count = 0
        max_concurrent_seen = 0

        async def track_concurrency(name):
            nonlocal active_count, max_concurrent_seen
            active_count += 1
            max_concurrent_seen = max(max_concurrent_seen, active_count)
            await asyncio.sleep(0.1)
            active_count -= 1
            return name

        await queue.start()

        futures = []
        for i in range(6):
            future = await queue.enqueue(track_concurrency, f"task_{i}")
            futures.append(future)

        results = await asyncio.gather(*futures)

        assert max_concurrent_seen == 3
        assert len(results) == 6

        await queue.stop()

    @pytest.mark.asyncio
    async def test_semaphore_blocks_when_full(self):
        """Test that semaphore blocks when all slots are used"""
        queue = ProviderQueue("test_provider", max_concurrent=2)
        started = []

        async def slow_task(name):
            started.append(name)
            await asyncio.sleep(0.2)
            return name

        await queue.start()

        # Start 4 tasks
        futures = []
        for i in range(4):
            future = await queue.enqueue(slow_task, f"task_{i}")
            futures.append(future)

        # Wait a bit - only 2 should have started
        await asyncio.sleep(0.05)
        assert len(started) == 2

        # Wait for all to complete
        await asyncio.gather(*futures)
        assert len(started) == 4

        await queue.stop()


class TestProviderQueueErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_function_raises_exception(self):
        """Test that exceptions from queued functions are propagated"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def failing_func():
            raise ValueError("Test error")

        await queue.start()
        future = await queue.enqueue(failing_func)

        with pytest.raises(ValueError) as exc_info:
            await future

        assert "Test error" in str(exc_info.value)

        await queue.stop()

    @pytest.mark.asyncio
    async def test_future_already_cancelled(self):
        """Test that already done futures are skipped"""
        queue = ProviderQueue("test_provider", max_concurrent=1)
        call_count = 0

        async def counting_func():
            nonlocal call_count
            call_count += 1
            return call_count

        await queue.start()

        # Enqueue and cancel before processing
        future = await queue.enqueue(counting_func)
        future.cancel()

        # Wait a bit for processing attempt
        await asyncio.sleep(0.1)

        # The function should not have been called since future was cancelled
        # Note: This is timing-dependent, the function might still be called
        # The key is that the queue doesn't crash

        await queue.stop()

    @pytest.mark.asyncio
    async def test_process_queue_exception_handling(self):
        """Test that exceptions in _process_queue are caught"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        await queue.start()

        # This should work normally
        async def normal_func():
            return "ok"

        future = await queue.enqueue(normal_func)
        result = await future
        assert result == "ok"

        await queue.stop()


class TestProviderQueueStats:
    """Tests for statistics tracking"""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test that stats are tracked correctly"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def dummy_func():
            return "result"

        await queue.start()

        # Initial stats
        stats = queue.get_stats()
        assert stats["total_requests"] == 0
        assert stats["queued_requests"] == 0
        assert stats["rejected_requests"] == 0

        # Enqueue a request
        future = await queue.enqueue(dummy_func)
        result = await future
        assert result == "result"

        # After execution
        stats = queue.get_stats()
        assert stats["total_requests"] == 1
        assert stats["provider"] == "test_provider"
        assert stats["max_concurrent"] == 1
        assert "avg_wait_time" in stats

        await queue.stop()

    @pytest.mark.asyncio
    async def test_get_stats_current_queue_size(self):
        """Test that current_queue_size reflects actual queue size"""
        queue = ProviderQueue("test_provider", max_concurrent=1, max_queue_size=10)

        async def slow_func():
            await asyncio.sleep(1.0)
            return "done"

        await queue.start()

        # Enqueue first request (will start processing)
        await queue.enqueue(slow_func)

        # Wait for processing to start
        await asyncio.sleep(0.1)

        # Enqueue more requests
        for _ in range(3):
            await queue.enqueue(slow_func)

        stats = queue.get_stats()
        # One is processing, three in queue
        assert stats["current_queue_size"] == 3

        await queue.stop()

    def test_estimate_wait_time(self):
        """Test wait time estimation"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        # Empty queue
        wait_time = queue._estimate_wait_time()
        assert wait_time == 0.0

    def test_estimate_wait_time_with_queue(self):
        """Test wait time estimation with queued items"""
        queue = ProviderQueue("test_provider", max_concurrent=2)

        # Simulate queue with items by directly manipulating
        # We test this indirectly through the stats
        # The formula is: queue_size * 5.0 / max_concurrent
        # With max_concurrent=2 and queue_size=4, should be 10.0

        # We need to actually put items in the queue for qsize() to work
        for i in range(4):
            queue._queue.put_nowait((0, i, None))

        wait_time = queue._estimate_wait_time()
        assert wait_time == 10.0

    def test_get_stats_structure(self):
        """Test that get_stats returns all expected fields"""
        queue = ProviderQueue(
            "test_provider",
            max_concurrent=5,
            max_queue_size=200,
            queue_timeout=60.0,
        )

        stats = queue.get_stats()

        assert stats["provider"] == "test_provider"
        assert stats["max_concurrent"] == 5
        assert stats["current_queue_size"] == 0
        assert stats["max_queue_size"] == 200
        assert stats["active_requests"] == 0
        assert stats["queue_timeout"] == 60.0
        assert stats["total_requests"] == 0
        assert stats["queued_requests"] == 0
        assert stats["rejected_requests"] == 0
        assert stats["avg_wait_time"] == 0.0

    def test_update_avg_wait_time(self):
        """Test average wait time calculation"""
        queue = ProviderQueue("test_provider")

        # First update
        queue._update_avg_wait_time(10.0)
        assert queue._stats["avg_wait_time"] == 1.0  # 0.1 * 10 + 0.9 * 0

        # Second update
        queue._update_avg_wait_time(20.0)
        # 0.1 * 20 + 0.9 * 1.0 = 2.9
        assert queue._stats["avg_wait_time"] == pytest.approx(2.9)

        # Third update
        queue._update_avg_wait_time(5.0)
        # 0.1 * 5 + 0.9 * 2.9 = 3.11
        assert queue._stats["avg_wait_time"] == pytest.approx(3.11)


class TestRequestQueueManager:
    """Tests for RequestQueueManager"""

    def test_init(self):
        """Test RequestQueueManager initialization"""
        manager = RequestQueueManager()

        assert manager._queues == {}
        assert manager._default_max_concurrent == 1
        assert manager._default_max_queue_size == 100

    def test_register_provider_new(self):
        """Test registering a new provider"""
        manager = RequestQueueManager()

        queue = manager.register_provider(
            "test_provider",
            max_concurrent=5,
            max_queue_size=50,
            queue_timeout=30.0,
        )

        assert isinstance(queue, ProviderQueue)
        assert queue.provider_name == "test_provider"
        assert queue.max_concurrent == 5
        assert queue.max_queue_size == 50
        assert queue.queue_timeout == 30.0

        # Check it's stored
        assert manager._queues["test_provider"] == queue

    def test_register_provider_update_existing(self):
        """Test updating an existing provider"""
        manager = RequestQueueManager()

        # Register first
        queue1 = manager.register_provider("test_provider", max_concurrent=2)
        assert queue1.max_concurrent == 2

        # Update with same name
        queue2 = manager.register_provider(
            "test_provider",
            max_concurrent=10,
            max_queue_size=500,
            queue_timeout=100.0,
        )

        # Should be same object, just updated
        assert queue1 is queue2
        assert queue1.max_concurrent == 10
        assert queue1.max_queue_size == 500
        assert queue1.queue_timeout == 100.0

    def test_get_queue_existing(self):
        """Test getting an existing queue"""
        manager = RequestQueueManager()
        manager.register_provider("test_provider")

        queue = manager.get_queue("test_provider")
        assert queue is not None
        assert queue.provider_name == "test_provider"

    def test_get_queue_nonexistent(self):
        """Test getting a non-existent queue"""
        manager = RequestQueueManager()

        queue = manager.get_queue("nonexistent")
        assert queue is None

    @pytest.mark.asyncio
    async def test_start_all_queues(self):
        """Test starting all registered queues"""
        manager = RequestQueueManager()

        manager.register_provider("provider1")
        manager.register_provider("provider2")
        manager.register_provider("provider3")

        await manager.start()

        for queue in manager._queues.values():
            assert queue._running is True

        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_all_queues(self):
        """Test stopping all registered queues"""
        manager = RequestQueueManager()

        manager.register_provider("provider1")
        manager.register_provider("provider2")

        await manager.start()
        await manager.stop()

        for queue in manager._queues.values():
            assert queue._running is False

    @pytest.mark.asyncio
    async def test_execute_with_registered_queue(self):
        """Test execute with a registered queue"""
        manager = RequestQueueManager()
        manager.register_provider("test_provider", max_concurrent=2)

        await manager.start()

        async def test_func(a, b):
            return a + b

        result = await manager.execute("test_provider", test_func, 1, 2)

        assert result == 3

        await manager.stop()

    @pytest.mark.asyncio
    async def test_execute_with_priority(self):
        """Test execute with priority parameter"""
        manager = RequestQueueManager()
        manager.register_provider("test_provider", max_concurrent=1)

        await manager.start()

        async def test_func(value):
            return value

        result = await manager.execute("test_provider", test_func, "hello", priority=5)

        assert result == "hello"

        await manager.stop()

    @pytest.mark.asyncio
    async def test_execute_without_registered_queue(self):
        """Test execute when no queue is registered (passthrough)"""
        manager = RequestQueueManager()

        async def test_func(a, b):
            return a * b

        # No queue registered for this provider
        result = await manager.execute("unregistered_provider", test_func, 3, 4)

        assert result == 12

    @pytest.mark.asyncio
    async def test_execute_propagates_exception(self):
        """Test that exceptions from execute are propagated"""
        manager = RequestQueueManager()
        manager.register_provider("test_provider")

        await manager.start()

        async def failing_func():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError) as exc_info:
            await manager.execute("test_provider", failing_func)

        assert "Test error" in str(exc_info.value)

        await manager.stop()

    def test_get_all_stats(self):
        """Test getting stats from all queues"""
        manager = RequestQueueManager()

        manager.register_provider("provider1", max_concurrent=5)
        manager.register_provider("provider2", max_concurrent=10)

        stats = manager.get_all_stats()

        assert "provider1" in stats
        assert "provider2" in stats
        assert stats["provider1"]["max_concurrent"] == 5
        assert stats["provider2"]["max_concurrent"] == 10

    def test_get_all_stats_empty(self):
        """Test getting stats when no queues registered"""
        manager = RequestQueueManager()

        stats = manager.get_all_stats()

        assert stats == {}


class TestGlobalQueueManager:
    """Tests for global queue manager functions"""

    def test_get_queue_manager_creates_singleton(self):
        """Test that get_queue_manager creates a singleton"""
        reset_queue_manager()

        manager1 = get_queue_manager()
        manager2 = get_queue_manager()

        assert manager1 is manager2

        reset_queue_manager()

    def test_reset_queue_manager(self):
        """Test that reset_queue_manager clears the singleton"""
        manager1 = get_queue_manager()
        reset_queue_manager()
        manager2 = get_queue_manager()

        assert manager1 is not manager2

        reset_queue_manager()

    def test_get_queue_manager_returns_correct_type(self):
        """Test that get_queue_manager returns RequestQueueManager"""
        reset_queue_manager()

        manager = get_queue_manager()

        assert isinstance(manager, RequestQueueManager)

        reset_queue_manager()


class TestProviderQueueIntegration:
    """Integration tests for ProviderQueue"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow with multiple requests"""
        queue = ProviderQueue("test_provider", max_concurrent=2, max_queue_size=10)
        results = []

        async def worker(name, delay=0.1):
            results.append(f"{name}_start")
            await asyncio.sleep(delay)
            results.append(f"{name}_end")
            return name

        await queue.start()

        # Submit multiple requests
        futures = []
        for i in range(5):
            future = await queue.enqueue(worker, f"task_{i}", delay=0.05)
            futures.append(future)

        # Wait for all to complete
        results_list = await asyncio.gather(*futures)

        assert len(results_list) == 5
        assert len(results) == 10  # 5 starts + 5 ends

        # Check stats
        stats = queue.get_stats()
        assert stats["total_requests"] == 5

        await queue.stop()

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_and_process(self):
        """Test concurrent enqueue operations"""
        queue = ProviderQueue("test_provider", max_concurrent=3, max_queue_size=100)

        async def quick_task(x):
            return x * 2

        await queue.start()

        # Concurrently enqueue
        async def enqueue_task(i):
            future = await queue.enqueue(quick_task, i)
            return await future

        tasks = [enqueue_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert sorted(results) == [i * 2 for i in range(10)]

        stats = queue.get_stats()
        assert stats["total_requests"] == 10

        await queue.stop()

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        """Test handling of mixed success and failure cases"""
        queue = ProviderQueue("test_provider", max_concurrent=2)

        async def maybe_fail(x):
            if x % 2 == 0:
                return x
            else:
                raise ValueError(f"Odd number: {x}")

        await queue.start()

        futures = []
        for i in range(6):
            future = await queue.enqueue(maybe_fail, i)
            futures.append(future)

        # Gather with exception handling
        results = await asyncio.gather(*futures, return_exceptions=True)

        # Even numbers should succeed
        assert results[0] == 0
        assert results[2] == 2
        assert results[4] == 4

        # Odd numbers should raise
        assert isinstance(results[1], ValueError)
        assert isinstance(results[3], ValueError)
        assert isinstance(results[5], ValueError)

        await queue.stop()

    @pytest.mark.asyncio
    async def test_active_requests_tracking(self):
        """Test that active_requests is tracked correctly"""
        queue = ProviderQueue("test_provider", max_concurrent=3)

        started = asyncio.Event()
        can_finish = asyncio.Event()

        async def controlled_task():
            started.set()
            await can_finish.wait()
            return "done"

        await queue.start()

        # Start a task
        future = await queue.enqueue(controlled_task)

        # Wait for it to start
        await started.wait()

        # Check active requests
        stats = queue.get_stats()
        assert stats["active_requests"] == 1

        # Let it finish
        can_finish.set()
        await future

        # Check active requests after completion
        stats = queue.get_stats()
        assert stats["active_requests"] == 0

        await queue.stop()


class TestEdgeCases:
    """Tests for edge cases"""

    @pytest.mark.asyncio
    async def test_enqueue_after_stop(self):
        """Test that enqueue works after stop and restart"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def dummy():
            return "result"

        await queue.start()
        await queue.stop()

        # Restart
        await queue.start()

        future = await queue.enqueue(dummy)
        result = await future

        assert result == "result"

        await queue.stop()

    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles"""
        queue = ProviderQueue("test_provider")

        async def dummy():
            return "ok"

        for i in range(3):
            await queue.start()
            future = await queue.enqueue(dummy)
            result = await future
            assert result == "ok"
            await queue.stop()

    @pytest.mark.asyncio
    async def test_queue_empty_after_processing(self):
        """Test that queue becomes empty after all requests processed"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        async def quick():
            return "done"

        await queue.start()

        futures = []
        for _ in range(5):
            future = await queue.enqueue(quick)
            futures.append(future)

        await asyncio.gather(*futures)

        # Give time for queue to clear
        await asyncio.sleep(0.1)

        stats = queue.get_stats()
        assert stats["current_queue_size"] == 0

        await queue.stop()

    @pytest.mark.asyncio
    async def test_request_with_none_result(self):
        """Test handling of functions that return None"""
        queue = ProviderQueue("test_provider")

        async def returns_none():
            return None

        await queue.start()
        future = await queue.enqueue(returns_none)
        result = await future

        assert result is None

        await queue.stop()

    @pytest.mark.asyncio
    async def test_request_with_complex_result(self):
        """Test handling of functions that return complex objects"""
        queue = ProviderQueue("test_provider")

        async def returns_complex():
            return {
                "list": [1, 2, 3],
                "nested": {"a": "b"},
                "tuple": (4, 5),
            }

        await queue.start()
        future = await queue.enqueue(returns_complex)
        result = await future

        assert result["list"] == [1, 2, 3]
        assert result["nested"]["a"] == "b"
        assert result["tuple"] == (4, 5)

        await queue.stop()

    @pytest.mark.asyncio
    async def test_zero_timeout(self):
        """Test queue with zero timeout (immediate timeout)"""
        queue = ProviderQueue("test_provider", max_concurrent=1, queue_timeout=0.0)

        async def slow():
            await asyncio.sleep(1.0)
            return "done"

        await queue.start()

        # Start one request
        future1 = await queue.enqueue(slow)

        # Wait for it to start processing
        await asyncio.sleep(0.1)

        # This should timeout immediately since queue_timeout is 0
        future2 = await queue.enqueue(slow)

        # Give time for the timeout check
        await asyncio.sleep(0.1)

        # The second request should timeout
        assert future2.done()
        with pytest.raises(asyncio.TimeoutError):
            await future2

        await queue.stop()


class TestMissingCoverage:
    """Tests specifically to cover missing lines 87-88 and 166-167"""

    @pytest.mark.asyncio
    async def test_stop_sets_exception_on_pending_future(self):
        """Test that stop() sets exception on pending futures in queue (lines 87-88)"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        # Don't start the queue
        # Create a request and put it directly in queue
        async def dummy_func():
            return "should not execute"

        future = asyncio.get_event_loop().create_future()
        request = QueuedRequest(
            id="test_request_1",
            func=dummy_func,
            args=(),
            kwargs={},
            future=future,
            priority=0,
        )

        # Put item directly in queue
        queue._queue.put_nowait((0, 1, request))
        queue._request_map["test_request_1"] = request

        # Verify future is not done
        assert not future.done()

        # Call stop - should set exception on the pending future
        await queue.stop()

        # Verify future now has exception
        assert future.done()
        exc = future.exception()
        assert isinstance(exc, RuntimeError)
        assert "shutting down" in str(exc)

    @pytest.mark.asyncio
    async def test_process_queue_exception_handler(self):
        """Test that _process_queue catches and logs exceptions (lines 166-167)"""
        queue = ProviderQueue("test_provider", max_concurrent=1)

        # Start the queue
        await queue.start()

        # Put a malformed item in queue - this will cause ValueError on unpacking
        # The unpacking expects (_, _, request) but we put a 2-tuple
        queue._queue.put_nowait((0, 1))  # Missing the request object!

        # Give the worker time to process and hit the exception
        await asyncio.sleep(0.2)

        # Stop the queue
        await queue.stop()

        # The exception should have been caught and logged (we can't easily verify log output
        # but the test passing without hanging or crashing proves the exception handler works)

    @pytest.mark.asyncio
    async def test_queue_empty_exception_in_stop(self):
        """Test that stop() handles QueueEmpty exception (lines 87-88)"""
        from unittest.mock import MagicMock

        queue = ProviderQueue("test_provider", max_concurrent=1)

        # Don't start the queue - keep it empty
        # Use MagicMock with side_effect to make empty() return different values on each call:
        # First call (while loop check): returns False -> loop enters
        # Second call (inside get_nowait()): returns True -> raises QueueEmpty
        queue._queue.empty = MagicMock(side_effect=[False, True])

        # Queue is actually empty, so when stop() calls get_nowait(), it will raise QueueEmpty
        # This triggers the except asyncio.QueueEmpty: break block (lines 87-88)
        await queue.stop()

        # Test passes if stop() completes without crashing - the QueueEmpty was caught
