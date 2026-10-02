from __future__ import annotations

"""
会话持久化存储。

默认目录结构：
.xingclaw/sessions/<session_id>/
  - meta.json
  - context.jsonl
  - events.jsonl
  - journal.jsonl

``events.jsonl`` 也是 Trace 的正式存储。旧版本留下的 ``traces.jsonl``
仍然会被读取，但新写入不会再依赖那个独立文件。
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.types import Message

from .serde import message_from_dict, message_to_dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


class SessionStore:
    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.root = self.workspace_dir / ".xingclaw" / "sessions" / session_id
        self.meta_file = self.root / "meta.json"
        self.session_file = self.root / "session.jsonl"
        self.context_file = self.root / "context.jsonl"
        self.events_file = self.root / "events.jsonl"
        self.traces_file = self.root / "traces.jsonl"
        self.journal_file = self.root / "journal.jsonl"
        self._journal_lock = threading.RLock()
        self._journal_next_seq: int | None = None

    def ensure_initialized(self, *, model_id: str, provider: str, system_prompt: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.meta_file.exists():
            # 为已有会话补建新版本的 journal 文件，保持升级兼容。
            if not self.journal_file.exists():
                self.journal_file.write_text("", encoding="utf-8")
            return
        meta = {
            "session_id": self.session_id,
            "model_id": model_id,
            "provider": provider,
            "system_prompt": system_prompt,
            "leaf_id": None,
            "parent_session_id": None,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        self._write_meta(meta)
        if not self.session_file.exists():
            header = {
                "type": "session",
                "version": 1,
                "id": self.session_id,
                "timestamp": _utc_now_iso(),
                "cwd": str(self.workspace_dir.resolve()),
                "parent_session": None,
            }
            self.session_file.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        if not self.context_file.exists():
            self.context_file.write_text("", encoding="utf-8")
        if not self.events_file.exists():
            self.events_file.write_text("", encoding="utf-8")
        # Trace 从这里开始和普通运行事件一样写入 events.jsonl。
        # 不主动创建 traces.jsonl；如果旧会话已经有这个文件，读取逻辑
        # 仍会兼容它，因此升级不会丢失历史数据。
        if not self.journal_file.exists():
            self.journal_file.write_text("", encoding="utf-8")

    def touch_updated_at(self) -> None:
        if not self.meta_file.exists():
            return
        meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        meta["updated_at"] = _utc_now_iso()
        self._write_meta(meta)

    def _write_meta(self, meta: dict[str, Any]) -> None:
        """Write session metadata in one place so timestamps stay consistent."""
        meta["updated_at"] = _utc_now_iso()
        self._atomic_write_text(
            self.meta_file,
            json.dumps(meta, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        """Replace a file atomically so a crash cannot leave half a JSON file."""
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(text, encoding="utf-8")
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def read_meta(self) -> dict[str, Any] | None:
        if not self.meta_file.exists():
            return None
        return json.loads(self.meta_file.read_text(encoding="utf-8"))

    def append_context_message(self, message: Message) -> str:
        entry = {
            "ts": _utc_now_iso(),
            "message": message_to_dict(message),
        }
        with self.context_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        return self.append_session_message(message)

    def append_event(self, event: dict[str, Any]) -> None:
        with self.events_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self.touch_updated_at()

    def append_journal_entry(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """追加一个可恢复的、带序号的 durable journal 记录。

        journal 是操作生命周期的事实记录，不替代现有 context/session 文件；每条
        记录先写入并 ``fsync``，因此崩溃后即使最后一行只写了一半，前面的操作仍
        可以可靠恢复。读取方会跳过坏掉的尾行，而不是让整个会话无法打开。
        """

        self.root.mkdir(parents=True, exist_ok=True)
        with self._journal_lock:
            if self._journal_next_seq is None:
                self._journal_next_seq = max(
                    (int(item.get("seq", 0)) for item in self.load_journal()),
                    default=0,
                ) + 1
            sequence = self._journal_next_seq
            self._journal_next_seq += 1
            entry = {
                "seq": sequence,
                "ts": _utc_now_iso(),
                "session_id": self.session_id,
                "kind": str(kind),
                "operation_id": operation_id,
                "payload": dict(payload or {}),
            }
            line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
            with self.journal_file.open("a", encoding="utf-8") as fp:
                fp.write(line)
                fp.flush()
                os.fsync(fp.fileno())
        self.touch_updated_at()
        return entry

    def load_journal(self) -> list[dict[str, Any]]:
        """读取 journal，忽略空行、损坏行和非对象值。"""

        if not self.journal_file.exists():
            return []
        entries: list[dict[str, Any]] = []
        for raw in self.journal_file.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                # 进程崩溃可能留下未完成的最后一行；前面的记录仍有效。
                continue
            if isinstance(value, dict) and isinstance(value.get("seq"), int):
                entries.append(value)
        entries.sort(key=lambda item: int(item.get("seq", 0)))
        return entries

    def load_recovery_journal(self) -> list[dict[str, Any]]:
        """Recovery must never silently skip a damaged committed record."""
        if not self.journal_file.exists():
            return []
        entries = []
        for raw in self.journal_file.read_bytes().splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                raise ValueError("恢复日志尾部不完整，请人工检查后恢复；不会自动重跑工具")
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError("恢复日志损坏，请人工检查") from exc
            if not isinstance(value,dict) or not isinstance(value.get("seq"),int) or (entries and value["seq"] <= entries[-1]["seq"]):
                raise ValueError("恢复日志序号无效，请人工检查")
            entries.append(value)
        return entries

    def start_operation(
        self,
        operation_id: str,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """记录一个需要最终收尾的操作。"""

        return self.append_journal_entry(
            "operation_started",
            {"operation": operation, **(metadata or {})},
            operation_id=operation_id,
        )

    def finish_operation(
        self,
        operation_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """记录操作终态（succeeded/failed/aborted 等）。"""

        return self.append_journal_entry(
            "operation_finished",
            {"status": status, **(metadata or {})},
            operation_id=operation_id,
        )

    def append_operation_event(
        self,
        operation_id: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """记录操作过程中的可选进度事件。"""

        return self.append_journal_entry(
            "operation_progress",
            {"event": event, **(payload or {})},
            operation_id=operation_id,
        )

    def load_incomplete_operations(self) -> list[dict[str, Any]]:
        """返回已 started 但尚未写入 finished 的操作，供启动恢复/诊断使用。"""

        active: dict[str, dict[str, Any]] = {}
        for entry in self.load_journal():
            operation_id = entry.get("operation_id")
            if not isinstance(operation_id, str) or not operation_id:
                continue
            kind = entry.get("kind")
            if kind == "operation_started":
                active[operation_id] = entry
            elif kind == "operation_finished":
                active.pop(operation_id, None)
        return list(active.values())

    def append_trace_event(self, trace: dict[str, Any]) -> None:
        """Append one compact Trace as a first-class event.

        Trace is a summary produced from the in-memory AgentEvent stream.  It
        deliberately does not copy the conversation contents; the event only
        stores timing, token, cost, tool and error statistics.  The envelope
        keeps the query keys outside the payload so an event-log reader can
        filter by ``thread_id``/``run_id`` without understanding every Trace
        field.
        """

        # 保持旧 append_trace API 的容错性：调用方即使还没显式执行
        # ensure_initialized，也可以先写入一条 Trace 事件。
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_file.touch(exist_ok=True)
        self.append_event(
            {
                "type": "trace",
                "event_version": 1,
                "recorded_at": _utc_now_iso(),
                "thread_id": self.session_id,
                "run_id": trace.get("run_id"),
                "trace_id": trace.get("trace_id"),
                "operation_id": trace.get("operation_id"),
                "trace": dict(trace),
            }
        )

    def append_trace(self, trace: dict[str, Any]) -> None:
        """Backward-compatible alias for :meth:`append_trace_event`.

        Older extensions/tests called ``append_trace`` directly.  Keeping the
        method avoids an API break while making even those writes use the new
        canonical events.jsonl storage.
        """

        self.append_trace_event(trace)

    def replace_trace(self, trace: dict[str, Any]) -> None:
        """Append a newer revision of a Trace to the event log.

        The old implementation rewrote ``traces.jsonl`` in place.  Event logs
        are append-only, so a replacement is represented by another ``trace``
        event with the same ``run_id``.  Querying deduplicates revisions and
        returns the newest one.  The old record remains available in the raw
        log for audit/debugging.
        """

        self.append_trace_event(trace)

    @staticmethod
    def _validate_thread_id(thread_id: str) -> str:
        """Validate a thread id before using it as a session directory name."""

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if thread_id in {".", ".."} or "/" in thread_id or "\\" in thread_id:
            raise ValueError("thread_id must be a single session id, not a path")
        return thread_id

    @staticmethod
    def _trace_key(trace: dict[str, Any], fallback: int) -> str:
        """Return the identity used to collapse append-only Trace revisions.

        ``run_id`` has priority because retries can share one logical
        ``operation_id`` while each attempt has its own run.  Falling back to
        ``trace_id``/``operation_id`` keeps old records queryable.
        """

        for field in ("run_id", "trace_id", "operation_id"):
            value = trace.get(field)
            if isinstance(value, str) and value:
                return f"{field}:{value}"
        return f"anonymous:{fallback}"

    def _load_trace_entries(self) -> list[tuple[dict[str, Any], str | None]]:
        """Read legacy and canonical Trace records in append order.

        The tuple contains ``(trace, thread_id)``.  Legacy records have no
        envelope, so they are associated with this SessionStore's session.
        Damaged JSONL lines are ignored just like the journal reader does.
        """

        entries: list[tuple[dict[str, Any], str | None]] = []

        # Read old data first.  Canonical events are then treated as newer,
        # which lets a new revision supersede a record from traces.jsonl.
        if self.traces_file.exists():
            for raw in self.traces_file.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    entries.append((value, self.session_id))

        if self.events_file.exists():
            for raw in self.events_file.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict) or value.get("type") != "trace":
                    continue

                payload = value.get("trace")
                if not isinstance(payload, dict):
                    continue
                trace = dict(payload)
                # The envelope is authoritative for routing/query fields.  A
                # hand-written event may omit them from the nested payload, so
                # fill only missing values rather than overwriting the payload.
                if not trace.get("run_id") and isinstance(value.get("run_id"), str):
                    trace["run_id"] = value["run_id"]
                if not trace.get("trace_id") and isinstance(value.get("trace_id"), str):
                    trace["trace_id"] = value["trace_id"]
                if not trace.get("operation_id") and isinstance(value.get("operation_id"), str):
                    trace["operation_id"] = value["operation_id"]
                thread_id = value.get("thread_id")
                if not isinstance(thread_id, str) or not thread_id:
                    nested_thread_id = trace.get("session_id")
                    thread_id = nested_thread_id if isinstance(nested_thread_id, str) else self.session_id
                entries.append((trace, thread_id))

        return entries

    def query_traces(
        self,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        operation_id: str | None = None,
        trace_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query Trace history from the event log, newest first.

        ``thread_id`` selects the session directory.  When it differs from
        this store's session, only that session's ``events.jsonl`` is read; no
        conversation is switched or mutated.  ``run_id`` is the exact
        low-level Agent run id.  ``operation_id`` is useful for viewing all
        retry attempts belonging to one user operation.
        """

        target_thread_id = self._validate_thread_id(thread_id or self.session_id)
        target = self if target_thread_id == self.session_id else SessionStore(self.workspace_dir, target_thread_id)

        latest: dict[str, dict[str, Any]] = {}
        for index, (trace, record_thread_id) in enumerate(target._load_trace_entries()):
            if record_thread_id not in {None, target_thread_id}:
                continue
            if run_id is not None and trace.get("run_id") != run_id:
                continue
            if operation_id is not None and trace.get("operation_id") != operation_id:
                continue
            if trace_id is not None and trace.get("trace_id") != trace_id:
                continue

            key = target._trace_key(trace, index)
            # Pop/reinsert moves a replacement to the newest position while
            # still leaving exactly one result for each run_id.
            latest.pop(key, None)
            latest[key] = trace

        traces = list(reversed(list(latest.values())))
        if limit is not None and limit >= 0:
            return traces[:limit]
        return traces

    def get_trace(
        self,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        operation_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest Trace matching the supplied identifiers."""

        traces = self.query_traces(
            thread_id=thread_id,
            run_id=run_id,
            operation_id=operation_id,
            trace_id=trace_id,
            limit=1,
        )
        return traces[0] if traces else None

    def load_traces(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Load this session's Trace history, newest first.

        Kept as the old convenience API; new callers that need filtering
        should use :meth:`query_traces`.
        """

        return self.query_traces(thread_id=self.session_id, limit=limit)

    def rewrite_context_messages(self, messages: list[Message]) -> None:
        self._rewrite_context_file(messages)
        self.touch_updated_at()
        self.rewrite_session_messages(messages)

    def _rewrite_context_file(self, messages: list[Message]) -> None:
        lines = [
            json.dumps({"ts": _utc_now_iso(), "message": message_to_dict(msg)}, ensure_ascii=False)
            for msg in messages
        ]
        self._atomic_write_text(
            self.context_file,
            "\n".join(lines) + ("\n" if lines else ""),
        )

    def append_compaction_entry(
        self,
        summary_message: Message,
        retained_messages: list[Message],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """以一个 compaction 节点切换上下文，同时保留旧 session 树。

        旧消息不会被删除或重新分配 ID。新的压缩分支从当前 leaf 下追加一个
        ``compaction`` entry，再追加保留的 recent message 副本；恢复这条分支时，
        compaction entry 会把此前的消息重置为 summary，再接上 recent。这样既能
        让 provider 看到短上下文，也能从树上追溯压缩前的完整历史。
        """

        self.root.mkdir(parents=True, exist_ok=True)
        lines = self._read_session_lines()
        header = lines[0] if lines and lines[0].get("type") == "session" else {
            "type": "session",
            "version": 1,
            "id": self.session_id,
            "timestamp": _utc_now_iso(),
            "cwd": str(self.workspace_dir.resolve()),
            "parent_session": None,
        }
        existing = lines[1:] if lines and lines[0].get("type") == "session" else lines
        meta = self.read_meta() or {}
        parent_id = meta.get("leaf_id")
        compaction_id = self._new_entry_id()
        timestamp = _utc_now_iso()
        compaction = {
            "type": "compaction",
            "id": compaction_id,
            "parent_id": parent_id if isinstance(parent_id, str) else None,
            "timestamp": timestamp,
            "summary_message": message_to_dict(summary_message),
            "metadata": dict(metadata or {}),
        }

        appended: list[dict[str, Any]] = [compaction]
        parent = compaction_id
        retained_ids: list[str] = []
        for message in retained_messages:
            entry_id = self._new_entry_id()
            retained_ids.append(entry_id)
            appended.append(
                {
                    "type": "message",
                    "id": entry_id,
                    "parent_id": parent,
                    "timestamp": _utc_now_iso(),
                    "message": message_to_dict(message),
                }
            )
            parent = entry_id

        # context 文件只保存当前可发送分支；session 文件通过原子替换保留所有
        # 旧 entry，并把压缩分支接在旧 leaf 后面。
        self._rewrite_context_file([summary_message, *retained_messages])
        self._write_session_lines([header, *existing, *appended])
        meta["leaf_id"] = parent
        if "session_id" not in meta:
            meta["session_id"] = self.session_id
        self._write_meta(meta)
        return [compaction_id, *retained_ids]

    def load_context_messages(self) -> list[Message]:
        if not self.context_file.exists():
            return []

        out: list[Message] = []
        for line in self.context_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            message_data = payload.get("message", {})
            if isinstance(message_data, dict):
                out.append(message_from_dict(message_data))
        return out

    def _new_entry_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _read_session_lines(self) -> list[dict[str, Any]]:
        if not self.session_file.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.session_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                out.append(data)
        return out

    def _write_session_lines(self, lines: list[dict[str, Any]]) -> None:
        text = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
        self._atomic_write_text(self.session_file, text + ("\n" if text else ""))
        self.touch_updated_at()

    def append_session_message(self, message: Message) -> str:
        """
        将消息写入 session tree（线性 parent 链），便于后续 fork/switch/branch。
        """

        meta = self.read_meta() or {}
        parent_id = meta.get("leaf_id")
        entry_id = self._new_entry_id()
        entry = {
            "type": "message",
            "id": entry_id,
            "parent_id": parent_id,
            "timestamp": _utc_now_iso(),
            "message": message_to_dict(message),
        }

        # Normal conversation growth is append-only.  Re-reading and
        # rewriting the whole session.jsonl made each message O(n), turning a
        # long session into O(n²) disk work.  Compaction/fork still use the
        # explicit rewrite methods below.
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.session_file.exists() or self.session_file.stat().st_size == 0:
            header = {
                "type": "session",
                "version": 1,
                "id": self.session_id,
                "timestamp": _utc_now_iso(),
                "cwd": str(self.workspace_dir.resolve()),
                "parent_session": None,
            }
            self.session_file.write_text(
                json.dumps(header, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        with self.session_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fp.flush()
            os.fsync(fp.fileno())

        meta["leaf_id"] = entry_id
        if "session_id" not in meta:
            meta["session_id"] = self.session_id
        self._write_meta(meta)
        return entry_id

    def rewrite_session_messages(self, messages: list[Message]) -> None:
        lines = self._read_session_lines()
        header = lines[0] if lines and lines[0].get("type") == "session" else {
            "type": "session",
            "version": 1,
            "id": self.session_id,
            "timestamp": _utc_now_iso(),
            "cwd": str(self.workspace_dir.resolve()),
            "parent_session": None,
        }
        rebuilt: list[dict[str, Any]] = [header]
        parent_id: str | None = None
        for message in messages:
            entry_id = self._new_entry_id()
            rebuilt.append(
                {
                    "type": "message",
                    "id": entry_id,
                    "parent_id": parent_id,
                    "timestamp": _utc_now_iso(),
                    "message": message_to_dict(message),
                }
            )
            parent_id = entry_id
        self._write_session_lines(rebuilt)
        meta = self.read_meta() or {}
        meta["leaf_id"] = parent_id
        if "session_id" not in meta:
            meta["session_id"] = self.session_id
        self._write_meta(meta)

    def load_session_messages(self, *, leaf_id: str | None = None) -> list[Message]:
        """
        从 session tree 恢复当前分支消息（默认使用 meta.leaf_id）。
        """

        lines = self._read_session_lines()
        if not lines:
            return []
        entries = [
            line
            for line in lines
            if line.get("type") in {"message", "compaction"}
        ]
        if not entries:
            return []
        by_id = {str(e.get("id")): e for e in entries if isinstance(e.get("id"), str)}
        meta = self.read_meta() or {}
        current = leaf_id or meta.get("leaf_id")
        if not isinstance(current, str) or current not in by_id:
            # 回退到最后一条，兼容旧数据
            current = str(entries[-1].get("id"))

        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            entry = by_id[current]
            chain.append(entry)
            parent_id = entry.get("parent_id")
            current = parent_id if isinstance(parent_id, str) else None

        chain.reverse()
        messages: list[Message] = []
        for entry in chain:
            if entry.get("type") == "compaction":
                # compaction 是上下文重置点：压缩前的历史仍保留在树上，但
                # 当前分支恢复时从 summary 重新开始。
                summary_data = entry.get("summary_message")
                if isinstance(summary_data, dict):
                    messages = [message_from_dict(summary_data)]
                continue
            msg_data = entry.get("message")
            if isinstance(msg_data, dict):
                messages.append(message_from_dict(msg_data))
        return messages

    def list_entry_ids(self) -> list[str]:
        lines = self._read_session_lines()
        return [
            str(line.get("id"))
            for line in lines
            if line.get("type") in {"message", "compaction"} and isinstance(line.get("id"), str)
        ]

    def get_leaf_id(self) -> str | None:
        meta = self.read_meta() or {}
        leaf = meta.get("leaf_id")
        return leaf if isinstance(leaf, str) else None

    def list_entries(self) -> list[dict[str, Any]]:
        """
        返回扁平 entry 列表，并补充导航信息：
        - depth: 根深度为 0
        - is_leaf: 是否当前叶子
        - preview: 文本摘要
        """

        lines = self._read_session_lines()
        entries = [
            line
            for line in lines
            if line.get("type") in {"message", "compaction"}
        ]
        by_id: dict[str, dict[str, Any]] = {}
        for entry in entries:
            eid = entry.get("id")
            if isinstance(eid, str):
                by_id[eid] = entry

        leaf_id = self.get_leaf_id()
        parent_by_id = {
            eid: entry.get("parent_id")
            for eid, entry in by_id.items()
        }
        depth_cache: dict[str, int] = {}

        def depth_for(entry_id: str) -> int:
            cached = depth_cache.get(entry_id)
            if cached is not None:
                return cached

            path: list[str] = []
            current = entry_id
            seen: set[str] = set()
            while current not in depth_cache and current not in seen:
                seen.add(current)
                path.append(current)
                parent = parent_by_id.get(current)
                if not isinstance(parent, str) or parent not in by_id:
                    # This node is a root (or points to a missing parent).
                    base_depth = -1
                    break
                current = parent
            else:
                # A cached ancestor gives us the base depth.  A malformed
                # cycle has no cached ancestor, so treat its boundary as root.
                base_depth = depth_cache.get(current, -1)

            for node_id in reversed(path):
                base_depth += 1
                depth_cache[node_id] = base_depth
            return depth_cache[entry_id]

        result: list[dict[str, Any]] = []
        for eid, entry in by_id.items():
            if entry.get("type") == "compaction":
                msg = entry.get("summary_message", {})
                role = "compaction"
            else:
                msg = entry.get("message", {})
                role = msg.get("role") if isinstance(msg, dict) else "unknown"
            result.append(
                {
                    "id": eid,
                    "parent_id": entry.get("parent_id"),
                    "timestamp": entry.get("timestamp"),
                    "role": role,
                    "preview": self._preview_message(msg if isinstance(msg, dict) else {}),
                    "depth": max(depth_for(eid), 0),
                    "is_leaf": eid == leaf_id,
                }
            )
        result.sort(key=lambda item: str(item.get("timestamp", "")))
        return result

    def get_entry_path(self, entry_id: str) -> list[str]:
        """
        返回从根到指定 entry 的 id 路径。
        """

        lines = self._read_session_lines()
        by_id = {
            str(line.get("id")): line
            for line in lines
            if line.get("type") in {"message", "compaction"} and isinstance(line.get("id"), str)
        }
        if entry_id not in by_id:
            raise ValueError(f"Entry not found: {entry_id}")

        path: list[str] = []
        current: str | None = entry_id
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            path.append(current)
            parent = by_id[current].get("parent_id")
            current = parent if isinstance(parent, str) else None
        path.reverse()
        return path

    def set_leaf(self, entry_id: str) -> None:
        lines = self._read_session_lines()
        ids = {
            str(line.get("id"))
            for line in lines
            if line.get("type") in {"message", "compaction"} and isinstance(line.get("id"), str)
        }
        if entry_id not in ids:
            raise ValueError(f"Entry not found: {entry_id}")
        meta = self.read_meta() or {}
        meta["leaf_id"] = entry_id
        self._write_meta(meta)

    def get_session_tree(self) -> list[dict[str, Any]]:
        """
        返回 session 树结构（按 parent_id 组织）。
        每个节点形如：
        {
          "id": "...",
          "parent_id": "...|None",
          "timestamp": "...",
          "role": "user|assistant|toolResult",
          "preview": "...",
          "children": [...]
        }
        """

        lines = self._read_session_lines()
        entries = [
            line
            for line in lines
            if line.get("type") in {"message", "compaction"}
        ]
        node_by_id: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []

        for entry in entries:
            eid = entry.get("id")
            if not isinstance(eid, str):
                continue
            if entry.get("type") == "compaction":
                msg = entry.get("summary_message", {})
                role = "compaction"
            else:
                msg = entry.get("message", {})
                role = msg.get("role") if isinstance(msg, dict) else "unknown"
            preview = self._preview_message(msg if isinstance(msg, dict) else {})
            node_by_id[eid] = {
                "id": eid,
                "parent_id": entry.get("parent_id"),
                "timestamp": entry.get("timestamp"),
                "role": role,
                "preview": preview,
                "children": [],
            }

        for node in node_by_id.values():
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id in node_by_id:
                node_by_id[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def _preview_message(message: dict[str, Any]) -> str:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            if isinstance(content, str):
                return content[:80]
            if isinstance(content, list):
                text = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += str(block.get("text", ""))
                return text[:80]
        if role == "assistant" and isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text", ""))
            return text[:80]
        if role == "toolResult" and isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text", ""))
            return text[:80]
        return ""

    def fork_to(
        self,
        new_session_id: str,
        *,
        from_entry_id: str | None = None,
    ) -> "SessionStore":
        """
        基于当前会话分支创建新会话（fork）。
        """

        target = SessionStore(self.workspace_dir, new_session_id)
        meta = self.read_meta() or {}
        target.ensure_initialized(
            model_id=str(meta.get("model_id", "")),
            provider=str(meta.get("provider", "")),
            system_prompt=str(meta.get("system_prompt", "")),
        )

        messages = self.load_session_messages(leaf_id=from_entry_id)
        target.rewrite_context_messages(messages)

        tmeta = target.read_meta() or {}
        tmeta["parent_session_id"] = self.session_id
        target._write_meta(tmeta)
        target.append_event(
            {
                "type": "session_forked",
                "from_session_id": self.session_id,
                "from_entry_id": from_entry_id,
                "to_session_id": new_session_id,
            }
        )
        return target
