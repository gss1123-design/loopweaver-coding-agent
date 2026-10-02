from __future__ import annotations

"""Durable IM inbox。

Webhook 可能在 worker 重启前后重复投递；仅把 message id 放在内存 set 里会让
重启后的重放再次触发 agent。这里用 append-only JSONL 记录状态迁移，启动时按
最后状态恢复尚未完成的消息。
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .types import IMIncomingMessage


class DurableInbox:
    """按 message_id 去重并恢复 pending/processing 入站消息。"""

    _PENDING_STATUSES = {"accepted", "processing"}
    _TERMINAL_STATUSES = {"completed", "dropped"}

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir) / ".xingclaw" / "im"
        self.path = self.root / "inbox.jsonl"
        self._lock = threading.RLock()
        self._state: dict[str, dict[str, Any]] = {}
        self._next_seq = 1
        self._load()

    def accept(self, message: IMIncomingMessage) -> bool:
        """持久化接收意图；返回 False 表示已接受过且仍在处理/已完成。"""

        message_id = str(message.message_id or "").strip()
        if not message_id:
            # 没有平台消息 ID 时无法可靠去重，但仍允许当前请求处理。
            return True
        with self._lock:
            previous = self._state.get(message_id)
            if previous and previous.get("status") in (*self._PENDING_STATUSES, *self._TERMINAL_STATUSES):
                return False
            self._append(
                message_id,
                "accepted",
                message=self._message_to_dict(message),
            )
            return True

    def mark_processing(self, message_id: str | None) -> None:
        self._mark(message_id, "processing")

    def mark_completed(self, message_id: str | None) -> None:
        self._mark(message_id, "completed")

    def mark_dropped(self, message_id: str | None, reason: str | None = None) -> None:
        self._mark(message_id, "dropped", reason=reason)

    def mark_failed(self, message_id: str | None, error: str | None = None) -> None:
        # failed 不属于 pending，后续同一个 id 的重新投递可以再次 accept。
        self._mark(message_id, "failed", error=error)

    def pending_messages(self) -> list[IMIncomingMessage]:
        """返回重启后需要重新排队的消息，按 inbox 序号顺序排列。"""

        with self._lock:
            records = [
                record
                for record in self._state.values()
                if record.get("status") in self._PENDING_STATUSES
            ]
        records.sort(key=lambda item: int(item.get("seq", 0)))
        messages: list[IMIncomingMessage] = []
        for record in records:
            payload = record.get("message")
            if isinstance(payload, dict):
                messages.append(self._message_from_dict(payload))
        return messages

    def status(self, message_id: str | None) -> str | None:
        if not message_id:
            return None
        record = self._state.get(str(message_id))
        value = record.get("status") if record else None
        return value if isinstance(value, str) else None

    def _mark(self, message_id: str | None, status: str, **extra: Any) -> None:
        if not message_id:
            return
        with self._lock:
            previous = self._state.get(str(message_id))
            if previous is None:
                return
            if "message" not in extra and isinstance(previous.get("message"), dict):
                extra["message"] = previous["message"]
            self._append(str(message_id), status, **extra)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                # 崩溃造成的半行不应阻断后续 webhook 恢复。
                continue
            if not isinstance(record, dict):
                continue
            message_id = record.get("message_id")
            if not isinstance(message_id, str) or not message_id:
                continue
            self._state[message_id] = record
            try:
                self._next_seq = max(self._next_seq, int(record.get("seq", 0)) + 1)
            except (TypeError, ValueError):
                continue

    def _append(self, message_id: str, status: str, **extra: Any) -> None:
        record = {
            "seq": self._next_seq,
            "ts": time.time(),
            "message_id": message_id,
            "status": status,
            **extra,
        }
        self._next_seq += 1
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        self._state[message_id] = record

    @staticmethod
    def _message_to_dict(message: IMIncomingMessage) -> dict[str, Any]:
        return {
            "platform": message.platform,
            "channel_id": message.channel_id,
            "user_id": message.user_id,
            "text": message.text,
            "thread_id": message.thread_id,
            "message_id": message.message_id,
            "created_at": message.created_at,
            "raw": dict(message.raw),
        }

    @staticmethod
    def _message_from_dict(payload: dict[str, Any]) -> IMIncomingMessage:
        raw = payload.get("raw")
        return IMIncomingMessage(
            platform=str(payload.get("platform", "")),
            channel_id=str(payload.get("channel_id", "")),
            user_id=str(payload.get("user_id", "")),
            text=str(payload.get("text", "")),
            thread_id=payload.get("thread_id") if isinstance(payload.get("thread_id"), str) else None,
            message_id=payload.get("message_id") if isinstance(payload.get("message_id"), str) else None,
            created_at=payload.get("created_at") if isinstance(payload.get("created_at"), (int, float)) else None,
            raw=dict(raw) if isinstance(raw, dict) else {},
        )
