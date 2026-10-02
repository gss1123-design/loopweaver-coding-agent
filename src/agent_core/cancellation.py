from __future__ import annotations

"""协作式取消原语。

``Agent.abort`` 可能从同步的 UI/IM 回调触发，而 agent loop、工具和 provider
运行在 asyncio 任务中。因此这里不用绑定单个事件循环的 ``asyncio.Event``，而是
用线程安全的 ``threading.Event`` 保存状态，再为等待者按需创建 loop-local event。

除了 LoopWeaver 自己的 :class:`CancellationToken`，辅助函数也接受具备常见
``throw_if_cancelled`` / ``is_set`` / ``is_cancelled`` 接口的外部 signal，方便
工具和 provider 渐进式接入。
"""

import asyncio
import inspect
import threading
from collections.abc import Awaitable
from typing import Any, TypeVar


_T = TypeVar("_T")


class CancellationToken:
    """线程安全、可传递的协作式取消令牌。

    ``cancel`` 只设置取消状态，不会强行中断正在运行的 Python 代码；调用方应在
    合适的边界调用 :meth:`throw_if_cancelled`，或者等待 :meth:`wait` 结束。这种
    语义比直接 ``Task.cancel`` 更适合工具、审批和 provider 共享同一个取消原因。
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._async_events: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Event]] = {}

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        """提供与 ``threading.Event`` / ``asyncio.Event`` 相似的兼容接口。"""

        return self._event.is_set()

    def cancelled(self) -> bool:
        """提供与 asyncio cancellation token 常见实现相似的兼容接口。"""

        return self.is_cancelled

    def cancel(self) -> bool:
        """标记取消，并唤醒所有正在 ``wait`` 的 asyncio 任务。

        返回值表示本次是否首次设置取消状态。
        """

        with self._lock:
            if self._event.is_set():
                return False
            self._event.set()
            waiters = list(self._async_events.values())

        for loop, event in waiters:
            if loop.is_closed():
                continue
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # loop 可能恰好在关闭，取消状态本身仍然已经可靠保存。
                continue
        return True

    def throw_if_cancelled(self) -> None:
        """若已取消，抛出 asyncio 的标准取消异常。"""

        if self.is_cancelled:
            raise asyncio.CancelledError

    async def wait(self) -> None:
        """异步等待取消，支持从另一个线程调用 ``cancel``。"""

        if self.is_cancelled:
            return

        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        with self._lock:
            waiter = self._async_events.get(loop_id)
            if waiter is None or waiter[0].is_closed():
                event = asyncio.Event()
                self._async_events[loop_id] = (loop, event)
            else:
                event = waiter[1]

            # cancel() 可能刚好在拿锁前完成；避免错过唤醒。
            already_cancelled = self._event.is_set()

        if already_cancelled:
            event.set()
        await event.wait()


def is_cancelled(signal: Any | None) -> bool:
    """读取外部 signal 的取消状态，兼容多种轻量接口。"""

    if signal is None:
        return False

    value = getattr(signal, "is_cancelled", False)
    if callable(value):
        value = value()
    if value:
        return True

    value = getattr(signal, "cancelled", False)
    if callable(value):
        value = value()
    if value:
        return True

    is_set = getattr(signal, "is_set", None)
    return bool(is_set()) if callable(is_set) else False


def throw_if_cancelled(signal: Any | None) -> None:
    """在 loop/tool/provider 边界检查取消状态。"""

    if signal is None:
        return

    throw = getattr(signal, "throw_if_cancelled", None)
    if callable(throw):
        throw()
        return

    if is_cancelled(signal):
        raise asyncio.CancelledError


async def await_with_cancellation(
    awaitable: Awaitable[_T] | _T,
    signal: Any | None,
) -> _T:
    """等待一个 awaitable，同时让支持 ``wait()`` 的 signal 可中断它。

    没有异步 ``wait`` 接口的外部 signal 会退化为普通 await；调用方仍可在
    await 前后调用 :func:`throw_if_cancelled`。被取消的业务 awaitable 会被清理，
    避免审批或 provider 任务在后台泄漏。
    """

    try:
        throw_if_cancelled(signal)
    except BaseException:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise
    if not inspect.isawaitable(awaitable):
        return awaitable

    wait = getattr(signal, "wait", None) if signal is not None else None
    if not callable(wait):
        return await awaitable

    operation = asyncio.ensure_future(awaitable)
    cancellation = asyncio.ensure_future(wait())
    try:
        done, _ = await asyncio.wait({operation, cancellation}, return_when=asyncio.FIRST_COMPLETED)
        if cancellation in done:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise asyncio.CancelledError
        cancellation.cancel()
        await asyncio.gather(cancellation, return_exceptions=True)
        return await operation
    finally:
        if not cancellation.done():
            cancellation.cancel()
        if not operation.done():
            operation.cancel()
        # gather the cancelled tasks so asyncio does not report them as leaked.
        pending: list[asyncio.Task[Any]] = [task for task in (operation, cancellation) if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
