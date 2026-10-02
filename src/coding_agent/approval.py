from __future__ import annotations

"""Human-in-the-loop approval primitives for tool execution."""

import asyncio
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolApprovalRequest:
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    requester_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "requester_id": self.requester_id,
        }


class ScopedApprovalGate:
    """Namespace child approvals while keeping decisions on the parent gate."""
    def __init__(self, parent: Any, scope: str):
        self.parent = parent
        self.scope = scope

    def id_for(self, tool_call_id: str) -> str:
        return f"{self.scope}:{tool_call_id}"

    def begin(self, request):
        return self.parent.begin(request)

    async def wait(self, tool_call_id):
        return await self.parent.wait(tool_call_id)

    def __getattr__(self, name):
        return getattr(self.parent, name)


class ApprovalGate:
    """Pause tool execution until the surrounding application decides.

    The gate is deliberately independent from CLI/IM code.  An integration
    can subscribe to the ``approval_required`` event, display the request, and
    call ``approve`` or ``reject`` using the tool call id.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 300.0,
        state_path: str | Path | None = None,
    ) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.state_path = Path(state_path) if state_path is not None else None
        self._pending: dict[str, tuple[ToolApprovalRequest, asyncio.Future[bool] | None]] = {}
        self._lock = threading.RLock()
        self._load_persisted()

    def begin(self, request: ToolApprovalRequest | dict[str, Any]) -> None:
        if isinstance(request, dict):
            request = ToolApprovalRequest(
                tool_call_id=str(request.get("tool_call_id", "")),
                tool_name=str(request.get("tool_name", "")),
                args=dict(request.get("args", {})),
                requester_id=str(request.get("requester_id", "") or ""),
            )
        loop = asyncio.get_running_loop()
        with self._lock:
            existing = self._pending.get(request.tool_call_id)
            if existing is not None and existing[1] is not None:
                return
            self._pending[request.tool_call_id] = (request, loop.create_future())
            self._persist_locked()

    def set_requester(self, tool_call_id: str, requester_id: str) -> bool:
        """Bind an approval to the user who initiated the surrounding task.

        Agent Core creates approval requests without knowing anything about the
        UI user.  The IM bridge fills that identity in when it receives the
        ``approval_required`` event.  Once set, the owner cannot be silently
        replaced by another user in the same channel.
        """

        requester = str(requester_id or "").strip()
        if not requester:
            return False
        with self._lock:
            item = self._pending.get(tool_call_id)
            if item is None:
                return False
            request, future = item
            if request.requester_id:
                return request.requester_id == requester
            self._pending[tool_call_id] = (
                ToolApprovalRequest(
                    tool_call_id=request.tool_call_id,
                    tool_name=request.tool_name,
                    args=dict(request.args),
                    requester_id=requester,
                ),
                future,
            )
            self._persist_locked()
            return True

    async def wait(self, tool_call_id: str) -> bool:
        with self._lock:
            item = self._pending.get(tool_call_id)
        if item is None:
            return False
        _, future = item
        if future is None:
            # 这是从上一次进程恢复的请求，但当前进程没有对应的等待任务；
            # 它仍可被 pending()/approve()/reject() 看见，不能伪造执行结果。
            return False
        try:
            return bool(await asyncio.wait_for(future, timeout=self.timeout_seconds))
        except asyncio.TimeoutError:
            return False
        finally:
            with self._lock:
                self._pending.pop(tool_call_id, None)
                self._persist_locked()

    def approve(self, tool_call_id: str) -> bool:
        return self.resolve(tool_call_id, True) == "approved"

    def reject(self, tool_call_id: str) -> bool:
        return self.resolve(tool_call_id, False) == "rejected"

    def resolve(
        self,
        tool_call_id: str,
        approved: bool,
        *,
        actor_id: str | None = None,
    ) -> str:
        """Resolve one request and return a machine-readable outcome.

        ``approve``/``reject`` remain as boolean compatibility wrappers.  IM
        integrations use this richer API so they can distinguish a genuine
        approval from an unauthorized user or a request restored after a
        process crash.  A restored request has no live Future, so claiming it
        was approved would be misleading: there is no suspended tool task left
        to wake up.
        """

        actor = str(actor_id or "").strip()
        with self._lock:
            item = self._pending.get(tool_call_id)
            if item is None:
                return "not_found"
            request, future = item
            if request.requester_id and actor and actor != request.requester_id:
                return "unauthorized"
            if future is None:
                self._pending.pop(tool_call_id, None)
                self._persist_locked()
                return "stale"
            if future.done():
                return "already_resolved"
            loop = future.get_loop()

        def _set_result() -> None:
            if not future.done():
                future.set_result(bool(approved))
            with self._lock:
                self._persist_locked()

        try:
            # Safe both on the owning event loop and from webhook worker
            # threads.  The waiting coroutine removes the item in wait().
            loop.call_soon_threadsafe(_set_result)
        except RuntimeError:
            # The old event loop has gone away.  Preserve the request so a
            # later caller can report it as stale rather than fake success.
            return "stale"
        return "approved" if approved else "rejected"

    def pending(self) -> list[dict[str, Any]]:
        # ``approve``/``reject`` resolve the Future first; the waiting tool
        # removes the entry in ``wait``'s finally block.  Hide already-resolved
        # requests immediately so UI/RPC status does not show stale approvals.
        with self._lock:
            pending: list[dict[str, Any]] = []
            for request, future in self._pending.values():
                if future is not None and future.done():
                    continue
                value = request.to_dict()
                value["recovered"] = future is None
                pending.append(value)
            return pending

    def reject_all(self) -> int:
        """Resolve all pending requests as rejected during shutdown/clear."""
        resolved = 0
        with self._lock:
            tool_call_ids = list(self._pending)
        for tool_call_id in tool_call_ids:
            if self.reject(tool_call_id):
                resolved += 1
        return resolved

    def _load_persisted(self) -> None:
        path = self.state_path
        if path is None or not path.exists():
            return
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(value, dict):
            value = value.get("pending", [])
        if not isinstance(value, list):
            return
        for raw in value:
            if not isinstance(raw, dict):
                continue
            tool_call_id = str(raw.get("tool_call_id") or "").strip()
            tool_name = str(raw.get("tool_name") or "").strip()
            if not tool_call_id or not tool_name:
                continue
            args = raw.get("args")
            self._pending[tool_call_id] = (
                ToolApprovalRequest(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    args=dict(args) if isinstance(args, dict) else {},
                    requester_id=str(raw.get("requester_id") or ""),
                ),
                None,
            )

    def _persist_locked(self) -> None:
        path = self.state_path
        if path is None:
            return
        pending = [request.to_dict() for request, future in self._pending.values() if future is None or not future.done()]
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps({"version": 1, "pending": pending}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with temp_path.open("a", encoding="utf-8") as fp:
                fp.flush()
                os.fsync(fp.fileno())
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
