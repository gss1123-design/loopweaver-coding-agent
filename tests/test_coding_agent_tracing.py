from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.types import AssistantMessage, Cost, TextContent, Usage
from ai.models import get_model
from coding_agent.agent_session import AgentSession
from coding_agent.session_store import SessionStore
from coding_agent.tracing import RunTraceRecorder, format_trace
from coding_agent.types import AgentSessionOptions


class CodingAgentTracingTests(unittest.TestCase):
    def test_recorder_builds_run_summary(self) -> None:
        recorder = RunTraceRecorder(session_id="s1")
        assistant = AssistantMessage(
            content=[TextContent(text="done")],
            provider="deepseek",
            model="deepseek-chat",
            usage=Usage(input=100, output=20, total_tokens=120, cost=Cost(total=0.01)),
            stop_reason="stop",
        )

        events = [
            {"type": "agent_start", "runId": "r1", "sessionId": "s1", "timestamp": 1000},
            {"type": "turn_start", "runId": "r1", "turnId": 1, "timestamp": 1001},
            {"type": "message_end", "runId": "r1", "turnId": 1, "timestamp": 1100, "message": assistant},
            {
                "type": "tool_execution_start",
                "runId": "r1",
                "turnId": 1,
                "timestamp": 1200,
                "toolCallId": "tc1",
                "toolName": "read_file",
            },
            {
                "type": "tool_execution_end",
                "runId": "r1",
                "turnId": 1,
                "timestamp": 1250,
                "toolCallId": "tc1",
                "toolName": "read_file",
                "isError": False,
            },
            {
                "type": "agent_end",
                "runId": "r1",
                "turnId": 1,
                "timestamp": 1300,
                "messages": [assistant],
            },
        ]

        completed = None
        for event in events:
            result = recorder.record_event(event)
            if result is not None:
                completed = result

        self.assertIsNotNone(completed)
        assert completed is not None
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["duration_ms"], 300)
        self.assertEqual(completed["turns"], 1)
        self.assertEqual(completed["tool_calls"], 1)
        self.assertEqual(completed["tool_errors"], 0)
        self.assertEqual(completed["input_tokens"], 100)
        self.assertEqual(completed["output_tokens"], 20)
        self.assertEqual(completed["total_tokens"], 120)
        self.assertEqual(completed["model"], "deepseek-chat")
        self.assertEqual(completed["tools"][0]["duration_ms"], 50)

    def test_recorder_marks_max_turns(self) -> None:
        recorder = RunTraceRecorder(session_id="s1")
        recorder.record_event({"type": "agent_start", "runId": "r1", "timestamp": 1000})
        recorder.record_event({"type": "max_turns_reached", "runId": "r1", "turns": 5, "timestamp": 2000})
        result = recorder.record_event({"type": "agent_end", "runId": "r1", "timestamp": 2100, "messages": []})

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "max_turns")
        self.assertEqual(result["turns"], 5)

    def test_recorder_builds_nested_spans_and_operation_metadata(self) -> None:
        recorder = RunTraceRecorder(session_id="s1")
        assistant = AssistantMessage(
            content=[TextContent(text="done")],
            provider="deepseek",
            model="deepseek-chat",
            usage=Usage(input=100, output=20, total_tokens=120, cost=Cost(total=0.01)),
            stop_reason="stop",
            response_id="resp-1",
        )
        events = [
            {
                "type": "agent_start",
                "runId": "r1",
                "operationId": "op-1",
                "attempt": 2,
                "sessionId": "s1",
                "timestamp": 1000,
            },
            {"type": "turn_start", "runId": "r1", "turnId": 1, "timestamp": 1001},
            {
                "type": "ai_request_start",
                "runId": "r1",
                "turnId": 1,
                "requestId": "req-1",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api": "openai-compatible",
                "streaming": True,
                "timestamp": 1010,
            },
            {
                "type": "ai_request_end",
                "runId": "r1",
                "turnId": 1,
                "requestId": "req-1",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api": "openai-compatible",
                "streaming": True,
                "timestamp": 1110,
                "durationMs": 100,
                "chunkCount": 3,
                "timeToFirstChunkMs": 20,
                "isError": False,
                "errorType": None,
                "responseId": "resp-1",
                "stopReason": "stop",
                "message": assistant,
            },
            {
                "type": "tool_execution_start",
                "runId": "r1",
                "turnId": 1,
                "timestamp": 1120,
                "toolCallId": "tc1",
                "toolName": "read_file",
            },
            {
                "type": "tool_execution_end",
                "runId": "r1",
                "turnId": 1,
                "timestamp": 1170,
                "toolCallId": "tc1",
                "toolName": "read_file",
                "isError": False,
            },
            {
                "type": "turn_end",
                "runId": "r1",
                "turnId": 1,
                "timestamp": 1180,
                "message": assistant,
                "toolResults": [],
            },
            {"type": "agent_end", "runId": "r1", "timestamp": 1200, "messages": [assistant]},
        ]

        result = None
        for event in events:
            result = recorder.record_event(event) or result

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["trace_id"], "op-1")
        self.assertEqual(result["operation_id"], "op-1")
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(result["ai_requests"], 1)
        self.assertEqual([span["kind"] for span in result["spans"]], ["run", "turn", "ai_request", "tool"])
        self.assertEqual(result["spans"][2]["time_to_first_chunk_ms"], 20)
        self.assertEqual(result["spans"][3]["turn_id"], 1)
        self.assertIn("ai_request", format_trace(result))
        self.assertIn("read_file", format_trace(result))

    def test_recorder_finishes_active_trace_without_agent_end(self) -> None:
        recorder = RunTraceRecorder(session_id="s1")
        recorder.record_event({"type": "agent_start", "runId": "r1", "timestamp": 1000})
        recorder.record_event({"type": "turn_start", "runId": "r1", "turnId": 1, "timestamp": 1001})

        result = recorder.finish_active(status="error", error="provider disconnected", timestamp=1050)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "error")
        self.assertEqual(result[0]["duration_ms"], 50)
        self.assertEqual(result[0]["spans"][0]["status"], "error")

    def test_session_recovers_crashed_operation_as_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.start_operation("op-crashed", "agent_run", {"max_attempts": 1})

            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                    session_id="s1",
                )
            )
            self.assertEqual(session.last_trace["status"], "crashed")  # type: ignore[index]
            self.assertEqual(session.last_trace["operation_id"], "op-crashed")  # type: ignore[index]
            session.close()

    def test_session_store_persists_compact_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.append_trace({"run_id": "r1", "status": "completed", "total_tokens": 10})
            store.append_trace({"run_id": "r2", "status": "error", "total_tokens": 20})

            traces = store.load_traces(limit=1)
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["run_id"], "r2")

    def test_trace_is_persisted_in_event_log_and_queryable_by_thread_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = SessionStore(workspace_dir=tmp_dir, session_id="thread-1")
            second = SessionStore(workspace_dir=tmp_dir, session_id="thread-2")
            first.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            second.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")

            first.append_trace_event(
                {
                    "run_id": "run-1",
                    "operation_id": "op-1",
                    "status": "completed",
                    "total_tokens": 10,
                }
            )
            first.append_trace_event(
                {
                    "run_id": "run-2",
                    "operation_id": "op-1",
                    "status": "error",
                    "total_tokens": 20,
                }
            )
            second.append_trace_event(
                {
                    "run_id": "run-other",
                    "operation_id": "op-other",
                    "status": "completed",
                    "total_tokens": 30,
                }
            )

            event_lines = [
                json.loads(line)
                for line in first.events_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([event["type"] for event in event_lines], ["trace", "trace"])
            self.assertEqual(event_lines[0]["thread_id"], "thread-1")
            self.assertEqual(event_lines[0]["run_id"], "run-1")
            self.assertEqual(event_lines[0]["trace"]["operation_id"], "op-1")
            self.assertFalse(first.traces_file.exists())

            self.assertEqual(first.query_traces(run_id="run-1")[0]["total_tokens"], 10)
            self.assertEqual(
                [item["run_id"] for item in first.query_traces(operation_id="op-1")],
                ["run-2", "run-1"],
            )
            self.assertEqual(
                first.query_traces(thread_id="thread-2")[0]["run_id"],
                "run-other",
            )

    def test_trace_revision_is_append_only_but_query_returns_latest_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.append_trace_event({"run_id": "r1", "status": "completed", "total_tokens": 10})
            store.replace_trace({"run_id": "r1", "status": "completed", "total_tokens": 25})

            traces = store.query_traces(run_id="r1")
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["total_tokens"], 25)
            event_lines = [
                json.loads(line)
                for line in store.events_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(event_lines), 2)
            self.assertEqual([event["type"] for event in event_lines], ["trace", "trace"])

    def test_legacy_traces_file_is_read_until_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.traces_file.write_text(
                json.dumps({"run_id": "legacy-run", "status": "completed"}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(store.query_traces(run_id="legacy-run")[0]["status"], "completed")
            store.append_trace_event({"run_id": "legacy-run", "status": "updated"})
            self.assertEqual(store.query_traces(run_id="legacy-run")[0]["status"], "updated")

    def test_agent_session_persists_trace_from_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                )
            )

            async def emit_run() -> None:
                await session._on_agent_event(
                    {"type": "agent_start", "runId": "r1", "timestamp": 1000}
                )
                await session._on_agent_event(
                    {"type": "turn_start", "runId": "r1", "turnId": 1, "timestamp": 1001}
                )
                await session._on_agent_event(
                    {"type": "agent_end", "runId": "r1", "timestamp": 1020, "messages": []}
                )

            asyncio.run(emit_run())
            self.assertEqual(session.last_trace["run_id"], "r1")  # type: ignore[index]
            self.assertEqual(session.store.load_traces(limit=1)[0]["duration_ms"], 20)
            session.close()

    def test_stream_deltas_are_not_written_as_one_event_per_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                )
            )

            async def emit_events() -> None:
                await session._on_agent_event(
                    {"type": "agent_start", "runId": "r1", "timestamp": 1000}
                )
                await session._on_agent_event(
                    {
                        "type": "message_update",
                        "runId": "r1",
                        "timestamp": 1001,
                        "assistantMessageEvent": {"type": "text_delta", "delta": "a"},
                    }
                )

            asyncio.run(emit_events())
            lines = session.store.events_file.read_text(encoding="utf-8").splitlines()
            persisted = [json.loads(line)["type"] for line in lines if line.strip()]
            self.assertEqual(persisted, ["agent_start"])
            session.close()


if __name__ == "__main__":
    unittest.main()
