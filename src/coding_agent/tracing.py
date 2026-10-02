from __future__ import annotations

"""Compact per-run tracing for coding-agent sessions.

The event stream already contains the detailed runtime events.  This module
turns those events into one small summary per agent run so callers can inspect
latency, token usage, tool failures, and the final status without reading the
full event log or storing prompt contents again.
"""

from dataclasses import dataclass, field
import json
import time
import uuid
from typing import Any

from ai.types import AssistantMessage


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return int(time.time() * 1000)


def _message_key(message: AssistantMessage) -> int:
    # A streamed assistant message appears in start/update/end events.  The
    # object identity lets us count its usage exactly once.
    return id(message)


@dataclass
class RunTrace:
    """A compact, JSON-serializable summary of one Agent run."""

    run_id: str
    session_id: str | None = None
    trace_id: str | None = None
    operation_id: str | None = None
    attempt: int = 1
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    status: str = "running"
    turns: int = 0
    event_count: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    ai_requests: int = 0
    external_spans: int = 0
    provider: str | None = None
    model: str | None = None
    last_stop_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> int | None:
        if self.started_at_ms is None or self.finished_at_ms is None:
            return None
        return max(0, self.finished_at_ms - self.started_at_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_version": 2,
            "trace_id": self.trace_id or self.run_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
            "attempt": self.attempt,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "turns": self.turns,
            "event_count": self.event_count,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "ai_requests": self.ai_requests,
            "external_spans": self.external_spans,
            "provider": self.provider,
            "model": self.model,
            "last_stop_reason": self.last_stop_reason,
            "errors": list(self.errors),
            "tools": [dict(item) for item in self.tools],
            "spans": [dict(item) for item in self.spans],
        }


class RunTraceRecorder:
    """Build run summaries from the existing AgentEvent stream."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self._active: dict[str, RunTrace] = {}
        self._counted_messages: dict[str, set[int]] = {}
        self.last_trace: dict[str, Any] | None = None

    @staticmethod
    def _new_span_id() -> str:
        return f"span_{uuid.uuid4().hex[:12]}"

    @classmethod
    def _start_span(
        cls,
        trace: RunTrace,
        *,
        kind: str,
        started_at_ms: int,
        parent_id: str | None,
        **fields: Any,
    ) -> dict[str, Any]:
        span = {
            "span_id": cls._new_span_id(),
            "parent_id": parent_id,
            "kind": kind,
            "started_at_ms": started_at_ms,
            "finished_at_ms": None,
            "duration_ms": None,
            "status": "running",
        }
        span.update({key: value for key, value in fields.items() if value is not None})
        trace.spans.append(span)
        return span

    @staticmethod
    def _finish_span(
        span: dict[str, Any],
        *,
        finished_at_ms: int,
        status: str,
        **fields: Any,
    ) -> None:
        if span.get("finished_at_ms") is not None:
            return
        started_at_ms = int(span.get("started_at_ms") or finished_at_ms)
        span["finished_at_ms"] = finished_at_ms
        span["duration_ms"] = max(0, finished_at_ms - started_at_ms)
        span["status"] = status
        span.update({key: value for key, value in fields.items() if value is not None})

    @staticmethod
    def _open_span(
        trace: RunTrace,
        *,
        kind: str,
        field: str | None = None,
        value: Any = None,
    ) -> dict[str, Any] | None:
        for span in reversed(trace.spans):
            if span.get("kind") != kind or span.get("finished_at_ms") is not None:
                continue
            if field is not None and span.get(field) != value:
                continue
            return span
        return None

    @classmethod
    def _finish_open_spans(cls, trace: RunTrace, *, timestamp: int, status: str) -> None:
        for span in trace.spans:
            if span.get("finished_at_ms") is None:
                cls._finish_span(span, finished_at_ms=timestamp, status=status)

    def finish_active(
        self,
        *,
        status: str = "crashed",
        error: str | None = None,
        timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """Finish runs that never emitted ``agent_end``.

        Normal provider failures usually emit an assistant error and an
        ``agent_end`` event.  This fallback covers exceptions, cancellation,
        and shutdown paths where the loop cannot emit its terminal event.
        """

        finished_at_ms = int(timestamp if timestamp is not None else time.time() * 1000)
        results: list[dict[str, Any]] = []
        for run_id, trace in list(self._active.items()):
            trace.finished_at_ms = finished_at_ms
            trace.status = status
            if error:
                error_text = str(error)[:500]
                if error_text and error_text not in trace.errors:
                    trace.errors.append(error_text)
            self._finish_open_spans(trace, timestamp=finished_at_ms, status=status)
            result = trace.to_dict()
            self.last_trace = result
            results.append(result)
            self._active.pop(run_id, None)
            self._counted_messages.pop(run_id, None)
        return results

    def add_external_span(
        self,
        *,
        kind: str,
        started_at_ms: int,
        finished_at_ms: int,
        status: str = "completed",
        parent_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Attach a completed non-loop span to the newest finished trace.

        Compaction happens between provider runs, so it has no active
        ``runId`` of its own.  Keeping it as a root span preserves the event in
        the same compact trace file without introducing a second trace store.
        """

        if self.last_trace is None:
            return None
        span = {
            "span_id": self._new_span_id(),
            "parent_id": parent_id,
            "kind": kind,
            "started_at_ms": started_at_ms,
            "finished_at_ms": finished_at_ms,
            "duration_ms": max(0, finished_at_ms - started_at_ms),
            "status": status,
        }
        span.update({key: value for key, value in fields.items() if value is not None})
        self.last_trace.setdefault("spans", []).append(span)
        self.last_trace["external_spans"] = int(self.last_trace.get("external_spans", 0) or 0) + 1
        return self.last_trace

    def record_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        run_id = event.get("runId")
        if not isinstance(run_id, str) or not run_id:
            return None

        timestamp = _timestamp_ms(event.get("timestamp"))
        trace = self._active.get(run_id)
        if trace is None:
            trace = RunTrace(
                run_id=run_id,
                session_id=event.get("sessionId") if isinstance(event.get("sessionId"), str) else self.session_id,
                trace_id=event.get("operationId") if isinstance(event.get("operationId"), str) else run_id,
                operation_id=event.get("operationId") if isinstance(event.get("operationId"), str) else None,
                attempt=int(event.get("attempt", 1) or 1),
                started_at_ms=timestamp,
            )
            self._active[run_id] = trace
            self._counted_messages[run_id] = set()
            self._ensure_run_span(trace, timestamp)
        elif isinstance(event.get("operationId"), str) and not trace.operation_id:
            trace.operation_id = event["operationId"]
            trace.trace_id = event["operationId"]
        if isinstance(event.get("attempt"), int):
            trace.attempt = max(1, int(event["attempt"]))

        trace.event_count += 1
        event_type = event.get("type")

        if event_type == "agent_start":
            trace.started_at_ms = timestamp
            self._ensure_run_span(trace, timestamp)
        elif event_type == "turn_start":
            turn_id = event.get("turnId")
            if isinstance(turn_id, int):
                trace.turns = max(trace.turns, turn_id)
            run_span = self._ensure_run_span(trace, timestamp)
            self._start_span(
                trace,
                kind="turn",
                started_at_ms=timestamp,
                parent_id=run_span.get("span_id"),
                turn_id=turn_id,
            )
        elif event_type == "turn_end":
            turn_id = event.get("turnId")
            span = self._open_span(trace, kind="turn", field="turn_id", value=turn_id)
            if span is not None:
                message = event.get("message")
                stop_reason = message.stop_reason if isinstance(message, AssistantMessage) else None
                self._finish_span(
                    span,
                    finished_at_ms=timestamp,
                    status="error" if stop_reason == "error" else ("aborted" if stop_reason == "aborted" else "completed"),
                    stop_reason=stop_reason,
                    tool_count=len(event.get("toolResults") or []),
                )
        elif event_type == "ai_request_start":
            self._record_ai_request_start(trace, event, timestamp)
        elif event_type == "ai_request_end":
            self._record_ai_request_end(trace, event, timestamp)
        elif event_type == "message_end":
            self._record_assistant_message(trace, event.get("message"), run_id)
        elif event_type == "tool_execution_start":
            self._record_tool_start(trace, event, timestamp)
        elif event_type == "tool_execution_end":
            self._record_tool_end(trace, event, timestamp)
        elif event_type == "error":
            error = str(event.get("error", "unknown error"))
            if error and error not in trace.errors:
                trace.errors.append(error[:500])
            trace.status = "error"
        elif event_type == "max_turns_reached":
            trace.status = "max_turns"
            turns = event.get("turns")
            if isinstance(turns, int):
                trace.turns = max(trace.turns, turns)
        elif event_type == "agent_end":
            # Some custom emitters omit message_end.  Use agent_end as a
            # fallback, while the identity set prevents double counting.
            for message in event.get("messages", []):
                self._record_assistant_message(trace, message, run_id)
            trace.finished_at_ms = timestamp
            if trace.status == "running":
                if trace.last_stop_reason == "error":
                    trace.status = "error"
                elif trace.last_stop_reason == "aborted":
                    trace.status = "aborted"
                else:
                    trace.status = "completed"
            self._finish_open_spans(trace, timestamp=timestamp, status=trace.status)
            result = trace.to_dict()
            self.last_trace = result
            self._active.pop(run_id, None)
            self._counted_messages.pop(run_id, None)
            return result

        return None

    def _ensure_run_span(self, trace: RunTrace, timestamp: int) -> dict[str, Any]:
        span = self._open_span(trace, kind="run")
        if span is not None:
            return span
        return self._start_span(trace, kind="run", started_at_ms=timestamp, parent_id=None)

    def _record_ai_request_start(self, trace: RunTrace, event: dict[str, Any], timestamp: int) -> None:
        trace.ai_requests += 1
        trace.provider = str(event.get("provider") or trace.provider or "") or trace.provider
        trace.model = str(event.get("model") or trace.model or "") or trace.model
        request_id = str(event.get("requestId") or f"request_{trace.ai_requests}")
        turn_span = self._open_span(trace, kind="turn")
        self._start_span(
            trace,
            kind="ai_request",
            started_at_ms=timestamp,
            parent_id=turn_span.get("span_id") if turn_span else self._ensure_run_span(trace, timestamp).get("span_id"),
            request_id=request_id,
            provider=event.get("provider"),
            model=event.get("model"),
            api=event.get("api"),
            streaming=bool(event.get("streaming", True)),
            turn_id=event.get("turnId"),
        )

    def _record_ai_request_end(self, trace: RunTrace, event: dict[str, Any], timestamp: int) -> None:
        request_id = str(event.get("requestId") or "")
        span = self._open_span(trace, kind="ai_request", field="request_id", value=request_id)
        if span is None:
            trace.ai_requests += 1
            trace.provider = str(event.get("provider") or trace.provider or "") or trace.provider
            trace.model = str(event.get("model") or trace.model or "") or trace.model
            span = self._start_span(
                trace,
                kind="ai_request",
                started_at_ms=timestamp,
                parent_id=self._ensure_run_span(trace, timestamp).get("span_id"),
                request_id=request_id,
            )
        message = event.get("message")
        if isinstance(message, AssistantMessage):
            self._record_assistant_message(trace, message, trace.run_id)
        error_type = event.get("errorType")
        status = (
            "error"
            if bool(event.get("isError"))
            or event.get("stopReason") in {"error", "aborted"}
            or error_type
            else "completed"
        )
        usage_fields: dict[str, Any] = {}
        if isinstance(message, AssistantMessage):
            usage_fields = {
                "input_tokens": int(message.usage.input or 0),
                "output_tokens": int(message.usage.output or 0),
                "total_tokens": int(message.usage.total_tokens or (message.usage.input + message.usage.output)),
                "cost": float(message.usage.cost.total or 0.0),
            }
        self._finish_span(
            span,
            finished_at_ms=timestamp,
            status=status,
            response_id=event.get("responseId"),
            stop_reason=event.get("stopReason"),
            chunk_count=event.get("chunkCount"),
            time_to_first_chunk_ms=event.get("timeToFirstChunkMs"),
            error_type=error_type,
            **usage_fields,
        )

    def _record_assistant_message(self, trace: RunTrace, message: Any, run_id: str) -> None:
        if not isinstance(message, AssistantMessage):
            return
        key = _message_key(message)
        counted = self._counted_messages.setdefault(run_id, set())
        if key in counted:
            return
        counted.add(key)

        usage = message.usage
        trace.input_tokens += int(usage.input or 0)
        trace.output_tokens += int(usage.output or 0)
        trace.total_tokens += int(usage.total_tokens or (usage.input + usage.output))
        trace.total_cost += float(usage.cost.total or 0.0)
        trace.provider = message.provider or trace.provider
        trace.model = message.model or trace.model
        trace.last_stop_reason = message.stop_reason

    def _record_tool_start(self, trace: RunTrace, event: dict[str, Any], timestamp: int) -> None:
        tool_call_id = str(event.get("toolCallId") or "")
        tool_name = str(event.get("toolName") or "unknown")
        trace.tool_calls += 1
        turn_span = self._open_span(trace, kind="turn")
        span = self._start_span(
            trace,
            kind="tool",
            started_at_ms=timestamp,
            parent_id=turn_span.get("span_id") if turn_span else self._ensure_run_span(trace, timestamp).get("span_id"),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            turn_id=event.get("turnId"),
        )
        trace.tools.append(
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "turn_id": event.get("turnId"),
                "span_id": span.get("span_id"),
                "started_at_ms": timestamp,
                "finished_at_ms": None,
                "duration_ms": None,
                "is_error": False,
            }
        )

    def _record_tool_end(self, trace: RunTrace, event: dict[str, Any], timestamp: int) -> None:
        tool_call_id = str(event.get("toolCallId") or "")
        is_error = bool(event.get("isError"))
        if is_error:
            trace.tool_errors += 1

        item = next(
            (entry for entry in reversed(trace.tools) if entry.get("tool_call_id") == tool_call_id and entry.get("finished_at_ms") is None),
            None,
        )
        if item is None:
            item = {
                "tool_call_id": tool_call_id,
                "tool_name": str(event.get("toolName") or "unknown"),
                "turn_id": event.get("turnId"),
                "started_at_ms": timestamp,
                "finished_at_ms": None,
                "duration_ms": None,
                "is_error": False,
            }
            trace.tools.append(item)
            trace.tool_calls += 1
        started = int(item.get("started_at_ms") or timestamp)
        item["finished_at_ms"] = timestamp
        item["duration_ms"] = max(0, timestamp - started)
        item["is_error"] = is_error
        span = self._open_span(trace, kind="tool", field="tool_call_id", value=tool_call_id)
        if span is None:
            span = self._start_span(
                trace,
                kind="tool",
                started_at_ms=started,
                parent_id=self._ensure_run_span(trace, started).get("span_id"),
                tool_call_id=tool_call_id,
                tool_name=event.get("toolName"),
                turn_id=event.get("turnId"),
            )
        item["span_id"] = span.get("span_id")
        self._finish_span(span, finished_at_ms=timestamp, status="error" if is_error else "completed", is_error=is_error)


def format_trace(trace: dict[str, Any] | None) -> str:
    """Render a compact, human-readable trace timeline for CLI/IM output."""

    if not trace:
        return "(no trace)"

    def duration(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "-"
        millis = max(0, int(value))
        if millis < 1000:
            return f"{millis}ms"
        return f"{millis / 1000:.2f}s"

    def status(value: Any) -> str:
        return str(value or "unknown")

    operation = trace.get("operation_id") or trace.get("trace_id") or trace.get("run_id") or "-"
    headline = (
        f"trace {operation} | {status(trace.get('status'))} | "
        f"duration={duration(trace.get('duration_ms'))} | "
        f"turns={trace.get('turns', 0)} | "
        f"tokens={trace.get('total_tokens', 0)} | "
        f"cost={float(trace.get('total_cost', 0.0) or 0.0):.6f}"
    )
    lines = [headline]
    attempt = trace.get("attempt")
    if attempt not in (None, 0, 1):
        lines.append(f"attempt: {attempt}")
    if trace.get("provider") or trace.get("model"):
        lines.append(f"model: {trace.get('provider') or '-'} / {trace.get('model') or '-'}")
    if trace.get("errors"):
        lines.append("errors: " + "; ".join(str(item) for item in trace["errors"][:3]))

    spans = [item for item in trace.get("spans", []) if isinstance(item, dict)]
    if not spans:
        # Keep the old JSON output for legacy summaries that predate spans;
        # newly recorded traces always have a timeline below the headline.
        if int(trace.get("trace_version", 1) or 1) < 2:
            return json.dumps(trace, ensure_ascii=False, indent=2, default=str)
        return "\n".join(lines)

    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for span in spans:
        by_parent.setdefault(span.get("parent_id"), []).append(span)
    for siblings in by_parent.values():
        siblings.sort(key=lambda item: int(item.get("started_at_ms") or 0))

    rendered: set[str] = set()

    def detail(span: dict[str, Any]) -> str:
        kind = span.get("kind")
        if kind == "ai_request":
            provider = span.get("provider") or "-"
            model = span.get("model") or "-"
            chunks = span.get("chunk_count")
            ttfb = span.get("time_to_first_chunk_ms")
            usage = span.get("total_tokens")
            return f" {provider}/{model} chunks={chunks or 0} ttfb={duration(ttfb)} tokens={usage or 0}"
        if kind == "tool":
            return f" {span.get('tool_name') or 'unknown'} error={bool(span.get('is_error'))}"
        if kind == "turn":
            return f" #{span.get('turn_id', '?')} tools={span.get('tool_count', 0)}"
        return ""

    def render(span: dict[str, Any], level: int) -> None:
        span_id = str(span.get("span_id") or "")
        if span_id in rendered:
            return
        rendered.add(span_id)
        lines.append(
            f"{'  ' * level}- {span.get('kind', 'span')} "
            f"{duration(span.get('duration_ms'))} [{status(span.get('status'))}]" + detail(span)
        )
        for child in by_parent.get(span_id, []):
            render(child, level + 1)

    roots = by_parent.get(None, [])
    for root in roots:
        render(root, 0)
    for span in spans:
        if str(span.get("span_id") or "") not in rendered:
            render(span, 0)
    return "\n".join(lines)


def format_trace_list(
    traces: list[dict[str, Any]],
    *,
    thread_id: str | None = None,
) -> str:
    """Render a compact index for querying Trace history.

    The detailed ``format_trace`` output is useful for one run.  A history
    query needs an index first, otherwise a user cannot tell which
    ``run_id`` to request next.
    """

    if not traces:
        prefix = f"thread_id={thread_id} " if thread_id else ""
        return f"{prefix}(no traces)"

    def duration(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "-"
        millis = max(0, int(value))
        return f"{millis}ms" if millis < 1000 else f"{millis / 1000:.2f}s"

    lines = [
        f"thread_id={thread_id or traces[0].get('session_id') or '-'} traces={len(traces)}"
    ]
    for trace in traces:
        lines.append(
            f"- run_id={trace.get('run_id') or '-'} "
            f"operation_id={trace.get('operation_id') or '-'} "
            f"status={trace.get('status') or 'unknown'} "
            f"duration={duration(trace.get('duration_ms'))} "
            f"turns={trace.get('turns', 0)} "
            f"tokens={trace.get('total_tokens', 0)}"
        )
    return "\n".join(lines)
