from __future__ import annotations

"""
统一事件流容器。

调用方可以：
1) async for 逐个消费事件；
2) await result() 获取最终 AssistantMessage。
"""

import asyncio
from typing import Any, AsyncIterator, Optional

from .types import AssistantMessage


_SENTINEL = object()


class AssistantMessageEventStream:
    def __init__(self, signal: Any | None = None) -> None:
        # 事件队列：用于迭代消费。
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        # 最终结果 Future：用于一次性获取完整消息。
        self._result: "asyncio.Future[AssistantMessage]" = asyncio.get_event_loop().create_future()
        self._closed = False
        self._signal = signal
        self._worker_task: asyncio.Task[Any] | None = None
        self._watch_task: asyncio.Task[Any] | None = None

        # provider 的后台任务可能正在等待网络 IO；只在 provider 自己的下一行
        # 检查 signal 仍然不够，所以这里额外监听异步取消令牌，及时关闭流并
        # 取消对应的 worker。没有 wait() 的外部 signal 仍由 provider 边界检查。
        if signal is not None:
            if _signal_is_cancelled(signal):
                self.cancel()
            else:
                wait = getattr(signal, "wait", None)
                if callable(wait):
                    loop = self._result.get_loop()
                    self._watch_task = loop.create_task(self._watch_signal(wait))

    async def _watch_signal(self, wait: Any) -> None:
        try:
            await wait()
        except asyncio.CancelledError:
            return
        self.cancel()

    def attach_task(self, task: asyncio.Task[Any]) -> None:
        """绑定 provider worker，以便 signal 取消时一起关闭网络请求。"""

        self._worker_task = task
        if self._closed and not task.done():
            task.cancel()

    def _stop_watcher(self) -> None:
        watcher = self._watch_task
        if watcher is None or watcher.done() or watcher is asyncio.current_task():
            return
        watcher.cancel()

    def cancel(self, error: BaseException | None = None) -> None:
        """以取消异常结束流，并中止 provider worker。"""

        if self._closed:
            return
        self._closed = True
        self._stop_watcher()
        if not self._result.done():
            if error is None:
                self._result.cancel()
            else:
                self._result.set_exception(error)
        self._queue.put_nowait(_SENTINEL)

        worker = self._worker_task
        if worker is not None and not worker.done() and worker is not asyncio.current_task():
            worker.cancel()

    def push(self, event: dict[str, Any]) -> None:
        """推送一个事件（text_delta/toolcall_delta/...）。"""
        if self._closed:
            return
        self._queue.put_nowait(event)

    def end(self, message: AssistantMessage) -> None:
        """正常结束：写入最终消息并关闭流。"""
        if self._closed:
            return
        self._closed = True
        self._stop_watcher()
        if not self._result.done():
            self._result.set_result(message)
        self._queue.put_nowait(_SENTINEL)

    def fail(self, error: BaseException, fallback: Optional[AssistantMessage] = None) -> None:
        """
        异常结束。

        fallback 存在时，result() 仍返回 fallback；
        否则 result() 抛出异常。
        """
        if self._closed:
            return
        self._closed = True
        self._stop_watcher()
        if fallback is not None:
            if not self._result.done():
                self._result.set_result(fallback)
        else:
            if not self._result.done():
                self._result.set_exception(error)
        self._queue.put_nowait(_SENTINEL)

    async def result(self) -> AssistantMessage:
        """等待并返回最终 AssistantMessage。"""
        return await self._result

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                break
            yield item


def _signal_is_cancelled(signal: Any) -> bool:
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
