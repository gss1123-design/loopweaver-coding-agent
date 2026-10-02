from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Callable

# 允许直接从源码目录导入（不依赖是否已 pip install -e .）
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.models import get_model
from ai.event_stream import AssistantMessageEventStream
from ai.types import AssistantMessage, TextContent, ToolCall, UserMessage
from agent_core.agent import Agent, AgentOptions
from agent_core.agent_loop import run_agent_loop, run_agent_loop_continue
from agent_core.cancellation import CancellationToken
from agent_core.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)
from coding_agent.approval import ApprovalGate
from coding_agent.builtin_tools import MUTATING_TOOL_NAMES, READ_ONLY_TOOL_NAMES, create_builtin_tools


class _FakeEventStream:
    def __init__(self, events: list[dict[str, Any]], final_message: AssistantMessage) -> None:
        self._events = events
        self._final = final_message
        self._idx = 0

    def __aiter__(self) -> "_FakeEventStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        item = self._events[self._idx]
        self._idx += 1
        return item

    async def result(self) -> AssistantMessage:
        return self._final


def _build_stream_fn(final_messages: list[AssistantMessage]) -> Callable[..., _FakeEventStream]:
    idx = {"value": 0}

    def _stream_fn(model: Any, context: Any, options: Any) -> _FakeEventStream:
        _ = model, context, options
        current = final_messages[idx["value"]]
        idx["value"] += 1
        return _FakeEventStream(events=[{"type": "done"}], final_message=current)

    return _stream_fn


def _make_config(**kwargs: Any) -> AgentLoopConfig:
    model = get_model("anthropic", "claude-sonnet-4-5")
    defaults: dict[str, Any] = {
        "model": model,
        "convert_to_llm": lambda messages: messages,
        "session_id": "test-session",
    }
    defaults.update(kwargs)
    return AgentLoopConfig(**defaults)


class AgentCoreLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_token_wakes_waiters(self) -> None:
        token = CancellationToken()
        waiting = asyncio.create_task(token.wait())
        await asyncio.sleep(0)

        self.assertTrue(token.cancel())
        await asyncio.wait_for(waiting, timeout=1)
        self.assertTrue(token.is_set())
        self.assertTrue(token.is_cancelled)
        self.assertFalse(token.cancel())

    async def test_cooperative_signal_reaches_tool_and_stops_loop(self) -> None:
        token = CancellationToken()
        tool_started = asyncio.Event()
        seen: dict[str, Any] = {}

        async def blocking_tool(
            tool_call_id: str,
            params: dict[str, Any],
            signal=None,
            on_update=None,
        ) -> AgentToolResult:
            _ = tool_call_id, params, on_update
            seen["tool_signal"] = signal
            tool_started.set()
            while True:
                signal.throw_if_cancelled()
                await asyncio.sleep(0.005)

        tool = AgentTool(
            name="blocking_tool",
            label="Blocking Tool",
            description="waits until cancelled",
            parameters={"type": "object", "properties": {}},
            execute=blocking_tool,
        )
        first = AssistantMessage(
            content=[ToolCall(id="tc-cancel", name="blocking_tool", arguments={})],
            stop_reason="toolUse",
        )
        second = AssistantMessage(content=[TextContent(text="never reached")], stop_reason="stop")

        def stream_fn(model: Any, context: Any, options: Any) -> _FakeEventStream:
            _ = model, context
            seen["provider_signal"] = options.signal
            return _build_stream_fn([first, second])(model, context, options)

        task = asyncio.create_task(
            run_agent_loop(
                prompts=[UserMessage(content="cancel this")],
                context=AgentContext(system_prompt="test", messages=[], tools=[tool]),
                config=_make_config(tool_execution="sequential"),
                emit=lambda event: None,
                signal=token,
                stream_fn=stream_fn,
            )
        )
        await asyncio.wait_for(tool_started.wait(), timeout=1)
        token.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        self.assertIs(seen["tool_signal"], token)
        self.assertIs(seen["provider_signal"], token)

    async def test_agent_abort_sets_signal_without_cancelling_run_task(self) -> None:
        model = get_model("anthropic", "claude-sonnet-4-5")
        agent = Agent(AgentOptions(model=model))
        started = asyncio.Event()
        seen: dict[str, Any] = {}

        async def fake_run_agent_loop(**kwargs: Any) -> list[Any]:
            signal = kwargs["signal"]
            seen["signal"] = signal
            started.set()
            while True:
                signal.throw_if_cancelled()
                await asyncio.sleep(0.005)

        from unittest.mock import patch

        with patch("agent_core.agent.run_agent_loop", new=fake_run_agent_loop):
            task = asyncio.create_task(agent.prompt("abort this"))
            await asyncio.wait_for(started.wait(), timeout=1)
            agent.abort()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)

        self.assertIsInstance(seen["signal"], CancellationToken)
        self.assertTrue(seen["signal"].is_cancelled)
        self.assertEqual(agent.state.error, "aborted")

    async def test_event_stream_cancellation_interrupts_result_wait(self) -> None:
        token = CancellationToken()
        stream = AssistantMessageEventStream(signal=token)
        result_task = asyncio.create_task(stream.result())
        await asyncio.sleep(0)

        token.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(result_task, timeout=1)

    async def test_retry_last_run_does_not_duplicate_prompt(self) -> None:
        model = get_model("anthropic", "claude-sonnet-4-5")
        agent = Agent(AgentOptions(model=model))
        failed = AssistantMessage(content=[], stop_reason="error", error_message="temporary outage")
        recovered = AssistantMessage(content=[TextContent(text="recovered")], stop_reason="stop")
        seen: dict[str, Any] = {}

        async def fake_run_agent_loop(**kwargs: Any) -> list[Any]:
            prompts = list(kwargs["prompts"])
            seen["prompts"] = prompts
            return [*prompts, failed]

        async def fake_run_agent_loop_continue(**kwargs: Any) -> list[Any]:
            seen["continue_context"] = list(kwargs["context"].messages)
            return [recovered]

        from unittest.mock import patch

        with patch("agent_core.agent.run_agent_loop", new=fake_run_agent_loop), patch(
            "agent_core.agent.run_agent_loop_continue", new=fake_run_agent_loop_continue
        ):
            first = await agent.prompt("hello")
            second = await agent.retry_last_run()

        self.assertEqual([getattr(message, "role", "") for message in first], ["user", "assistant"])
        self.assertEqual([getattr(message, "role", "") for message in second], ["assistant"])
        self.assertEqual(len(seen["prompts"]), 1)
        self.assertEqual(
            "".join(block.text for block in seen["prompts"][0].content if isinstance(block, TextContent)),
            "hello",
        )
        self.assertEqual([getattr(message, "role", "") for message in seen["continue_context"]], ["user"])
        self.assertEqual([getattr(message, "role", "") for message in agent.state.messages], ["user", "assistant"])
        self.assertEqual(
            "".join(block.text for block in agent.state.messages[0].content if isinstance(block, TextContent)),
            "hello",
        )

    async def test_request_scoped_system_prompt_does_not_mutate_agent_state(self) -> None:
        model = get_model("anthropic", "claude-sonnet-4-5")
        agent = Agent(AgentOptions(model=model, system_prompt="base prompt"))
        seen: dict[str, str] = {}

        async def fake_run_agent_loop(**kwargs: Any) -> list[Any]:
            seen["system_prompt"] = kwargs["context"].system_prompt
            return []

        from unittest.mock import patch

        with patch("agent_core.agent.run_agent_loop", new=fake_run_agent_loop):
            await agent.prompt("hello", system_prompt="base prompt\n\nactive skill")

        self.assertEqual(seen["system_prompt"], "base prompt\n\nactive skill")
        self.assertEqual(agent.state.system_prompt, "base prompt")

    def test_builtin_tools_declare_conservative_read_only_metadata(self) -> None:
        tools = create_builtin_tools(ROOT)
        by_name = {tool.name: tool for tool in tools}

        for name in READ_ONLY_TOOL_NAMES:
            self.assertIn(name, by_name)
            self.assertTrue(by_name[name].read_only, name)
            self.assertFalse(by_name[name].requires_approval, name)
        for name in MUTATING_TOOL_NAMES:
            self.assertIn(name, by_name)
            self.assertFalse(by_name[name].read_only, name)
            self.assertTrue(by_name[name].requires_approval, name)

    async def test_event_schema_contains_common_fields(self) -> None:
        events: list[AgentEvent] = []
        config = _make_config()
        context = AgentContext(system_prompt="test", messages=[], tools=[])
        prompt = UserMessage(content="hello")
        final = AssistantMessage(content=[TextContent(text="hi")], stop_reason="stop")

        new_messages = await run_agent_loop(
            prompts=[prompt],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([final]),
        )

        self.assertEqual(len(new_messages), 2)
        self.assertTrue(events)
        for e in events:
            self.assertIn("type", e)
            self.assertIn("runId", e)
            self.assertIn("turnId", e)
            self.assertIn("eventId", e)
            self.assertIn("timestamp", e)
            self.assertIn("sessionId", e)
            self.assertEqual(e["sessionId"], "test-session")

    async def test_events_carry_operation_attempt_and_ai_request_timing(self) -> None:
        events: list[AgentEvent] = []
        config = _make_config(operation_id="op-trace", attempt=3)
        context = AgentContext(system_prompt="test", messages=[], tools=[])
        final = AssistantMessage(content=[TextContent(text="hi")], stop_reason="stop")

        await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([final]),
        )

        self.assertTrue(events)
        self.assertTrue(all(e.get("operationId") == "op-trace" for e in events))
        self.assertTrue(all(e.get("attempt") == 3 for e in events))
        request_events = [e for e in events if e.get("type") in {"ai_request_start", "ai_request_end"}]
        self.assertEqual([e.get("type") for e in request_events], ["ai_request_start", "ai_request_end"])
        self.assertEqual(request_events[0].get("requestId"), request_events[1].get("requestId"))
        self.assertEqual(request_events[1].get("chunkCount"), 1)

    async def test_parallel_tool_execution(self) -> None:
        events: list[AgentEvent] = []
        tool_calls = [
            ToolCall(id="tc1", name="slow_tool", arguments={"path": "a.py"}),
            ToolCall(id="tc2", name="slow_tool", arguments={"path": "b.py"}),
        ]
        tool_started_at: list[float] = []

        async def slow_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
            _ = tool_call_id, params, signal, on_update
            tool_started_at.append(time.perf_counter())
            await asyncio.sleep(0.06)
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="slow_tool",
            label="Slow Tool",
            description="sleep",
            parameters={"type": "object", "properties": {}},
            execute=slow_tool,
            read_only=True,
        )

        first = AssistantMessage(content=tool_calls, stop_reason="toolUse")
        second = AssistantMessage(content=[TextContent(text="done")], stop_reason="stop")

        config = _make_config(tool_execution="parallel")
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])
        prompt = UserMessage(content="run tools")

        started = time.perf_counter()
        new_messages = await run_agent_loop(
            prompts=[prompt],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([first, second]),
        )
        elapsed = time.perf_counter() - started

        # 用开始时间差判断重叠，而不是用绝对耗时阈值；后者会受 CI/Windows
        # 调度抖动影响，导致并行实现偶发误报。
        self.assertEqual(len(tool_started_at), 2)
        self.assertLess(max(tool_started_at) - min(tool_started_at), 0.03)
        self.assertLess(elapsed, 0.20)
        tool_result_count = len([m for m in new_messages if getattr(m, "role", "") == "toolResult"])
        self.assertEqual(tool_result_count, 2)

        started_events = [e for e in events if e["type"] == "tool_execution_start"]
        self.assertEqual(len(started_events), 2)

    async def test_parallel_mode_falls_back_to_sequential_for_mutating_tools(self) -> None:
        active = 0
        max_active = 0

        async def mutating_tool(
            tool_call_id: str,
            params: dict[str, Any],
            signal=None,
            on_update=None,
        ) -> AgentToolResult:
            nonlocal active, max_active
            _ = tool_call_id, params, signal, on_update
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            return AgentToolResult(content=[TextContent(text="ok")])

        # read_only 默认 False：即使两个 path 不同，parallel 模式也必须保守降级。
        tool = AgentTool(
            name="mutating_tool",
            label="Mutating Tool",
            description="changes external state",
            parameters={"type": "object", "properties": {}},
            execute=mutating_tool,
        )
        first = AssistantMessage(
            content=[
                ToolCall(id="tc1", name="mutating_tool", arguments={"path": "a.py"}),
                ToolCall(id="tc2", name="mutating_tool", arguments={"path": "b.py"}),
            ],
            stop_reason="toolUse",
        )
        second = AssistantMessage(content=[TextContent(text="done")], stop_reason="stop")

        await run_agent_loop(
            prompts=[UserMessage(content="run mutating tools")],
            context=AgentContext(system_prompt="test", messages=[], tools=[tool]),
            config=_make_config(tool_execution="parallel"),
            emit=lambda event: None,
            stream_fn=_build_stream_fn([first, second]),
        )

        self.assertEqual(max_active, 1)

    async def test_parallel_mode_falls_back_to_sequential_for_overlapping_paths(self) -> None:
        active = 0
        max_active = 0

        async def read_only_tool(
            tool_call_id: str,
            params: dict[str, Any],
            signal=None,
            on_update=None,
        ) -> AgentToolResult:
            nonlocal active, max_active
            _ = tool_call_id, params, signal, on_update
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="read_only_tool",
            label="Read-only Tool",
            description="reads one path",
            parameters={"type": "object", "properties": {}},
            execute=read_only_tool,
            read_only=True,
        )
        first = AssistantMessage(
            content=[
                ToolCall(id="tc1", name="read_only_tool", arguments={"path": "src/app.py"}),
                ToolCall(id="tc2", name="read_only_tool", arguments={"path": "src/app.py"}),
            ],
            stop_reason="toolUse",
        )
        second = AssistantMessage(content=[TextContent(text="done")], stop_reason="stop")

        await run_agent_loop(
            prompts=[UserMessage(content="read the same path twice")],
            context=AgentContext(system_prompt="test", messages=[], tools=[tool]),
            config=_make_config(tool_execution="parallel"),
            emit=lambda event: None,
            stream_fn=_build_stream_fn([first, second]),
        )

        self.assertEqual(max_active, 1)

    async def test_before_hook_can_block_tool(self) -> None:
        events: list[AgentEvent] = []
        execute_count = {"value": 0}

        async def blocked_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
            _ = tool_call_id, params, signal, on_update
            execute_count["value"] += 1
            return AgentToolResult(content=[TextContent(text="should-not-run")])

        tool = AgentTool(
            name="dangerous_tool",
            label="Danger",
            description="blocked tool",
            parameters={"type": "object", "properties": {}},
            execute=blocked_tool,
        )

        async def before_hook(ctx, signal=None) -> BeforeToolCallResult:
            _ = ctx, signal
            return BeforeToolCallResult(block=True, reason="blocked-by-test")

        first = AssistantMessage(
            content=[ToolCall(id="tc_block", name="dangerous_tool", arguments={})],
            stop_reason="toolUse",
        )
        second = AssistantMessage(content=[TextContent(text="fallback")], stop_reason="stop")

        config = _make_config(tool_execution="sequential", before_tool_call=before_hook)
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        new_messages = await run_agent_loop(
            prompts=[UserMessage(content="try tool")],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([first, second]),
        )

        self.assertEqual(execute_count["value"], 0)
        blocked_results = [
            m
            for m in new_messages
            if getattr(m, "role", "") == "toolResult"
            and any(isinstance(c, TextContent) and "blocked-by-test" in c.text for c in m.content)
        ]
        self.assertTrue(blocked_results)
        end_events = [e for e in events if e["type"] == "tool_execution_end" and e.get("isError")]
        self.assertTrue(end_events)

    async def test_approval_gate_pauses_then_allows_tool(self) -> None:
        events: list[AgentEvent] = []
        execute_count = {"value": 0}
        gate = ApprovalGate(timeout_seconds=2)

        async def approved_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
            _ = tool_call_id, params, signal, on_update
            execute_count["value"] += 1
            return AgentToolResult(content=[TextContent(text="approved-result")])

        tool = AgentTool(
            name="approved_tool",
            label="Approved Tool",
            description="requires approval",
            parameters={"type": "object", "properties": {}},
            execute=approved_tool,
        )

        async def emit(event: AgentEvent) -> None:
            events.append(event)
            if event["type"] == "approval_required":
                self.assertEqual(event["toolName"], "approved_tool")
                self.assertTrue(gate.approve(event["toolCallId"]))

        first = AssistantMessage(
            content=[ToolCall(id="tc_approval", name="approved_tool", arguments={})],
            stop_reason="toolUse",
        )
        second = AssistantMessage(content=[TextContent(text="finished")], stop_reason="stop")
        config = _make_config(tool_execution="sequential", approval_gate=gate)
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        new_messages = await run_agent_loop(
            prompts=[UserMessage(content="use approved tool")],
            context=context,
            config=config,
            emit=emit,
            stream_fn=_build_stream_fn([first, second]),
        )

        self.assertEqual(execute_count["value"], 1)
        self.assertTrue(any(event["type"] == "approval_required" for event in events))
        self.assertTrue(any(getattr(message, "role", "") == "toolResult" for message in new_messages))

    async def test_approval_gate_skips_tool_marked_as_not_requiring_approval(self) -> None:
        events: list[AgentEvent] = []
        execute_count = {"value": 0}
        gate = ApprovalGate(timeout_seconds=0.05)

        async def read_tool(tool_call_id: str, params: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
            _ = tool_call_id, params, signal, on_update
            execute_count["value"] += 1
            return AgentToolResult(content=[TextContent(text="read-result")])

        tool = AgentTool(
            name="safe_read",
            label="Safe Read",
            description="read-only tool",
            parameters={"type": "object", "properties": {}},
            execute=read_tool,
            read_only=True,
            requires_approval=False,
        )
        first = AssistantMessage(
            content=[ToolCall(id="tc_safe_read", name="safe_read", arguments={})],
            stop_reason="toolUse",
        )
        second = AssistantMessage(content=[TextContent(text="finished")], stop_reason="stop")

        new_messages = await run_agent_loop(
            prompts=[UserMessage(content="read something")],
            context=AgentContext(system_prompt="test", messages=[], tools=[tool]),
            config=_make_config(tool_execution="sequential", approval_gate=gate),
            emit=events.append,
            stream_fn=_build_stream_fn([first, second]),
        )

        self.assertEqual(execute_count["value"], 1)
        self.assertFalse(any(event["type"] == "approval_required" for event in events))
        self.assertTrue(any(getattr(message, "role", "") == "toolResult" for message in new_messages))

    async def test_approval_gate_can_be_resolved_from_another_thread(self) -> None:
        gate = ApprovalGate(timeout_seconds=2)
        gate.begin({"tool_call_id": "tc-thread", "tool_name": "bash", "args": {}})
        waiting = asyncio.create_task(gate.wait("tc-thread"))
        await asyncio.sleep(0)

        resolved = await asyncio.to_thread(gate.approve, "tc-thread")
        self.assertTrue(resolved)
        self.assertTrue(await waiting)

    async def test_continue_loop(self) -> None:
        events: list[AgentEvent] = []
        context = AgentContext(
            system_prompt="test",
            messages=[UserMessage(content="continue please")],
            tools=[],
        )
        config = _make_config()
        final = AssistantMessage(content=[TextContent(text="continued")], stop_reason="stop")

        new_messages = await run_agent_loop_continue(
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_build_stream_fn([final]),
        )
        self.assertEqual(len(new_messages), 1)
        self.assertEqual(getattr(new_messages[0], "role", ""), "assistant")

        bad_context = AgentContext(
            system_prompt="test",
            messages=[AssistantMessage(content=[TextContent(text="already assistant")], stop_reason="stop")],
            tools=[],
        )
        with self.assertRaises(ValueError):
            await run_agent_loop_continue(
                context=bad_context,
                config=config,
                emit=events.append,
                stream_fn=_build_stream_fn([final]),
            )


if __name__ == "__main__":
    unittest.main()
