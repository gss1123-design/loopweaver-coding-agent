from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.types import AssistantMessage, TextContent, ToolCall
from agent_core import (
    AfterToolCallContext,
    AgentContext,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from coding_agent.factory import (
    _compose_after_tool_call,
    _compose_before_tool_call,
    create_agent_session,
)
from coding_agent.hooks import (
    create_read_only_tool_guard,
    create_secret_redaction_hook,
    create_workspace_path_guard,
)
from coding_agent.types import CreateAgentSessionOptions


def _tool(name: str = "custom", *, read_only: bool = False, properties: dict | None = None) -> AgentTool:
    return AgentTool(
        name=name,
        label=name,
        description="test tool",
        parameters={"type": "object", "properties": properties or {}},
        execute=lambda tool_call_id, params, signal=None, on_update=None: AgentToolResult(content=[]),
        read_only=read_only,
    )


def _before_context(tool: AgentTool, args: dict) -> BeforeToolCallContext:
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(),
        tool_call=ToolCall(id="tc-before", name=tool.name, arguments=args),
        args=args,
        context=AgentContext(system_prompt="", messages=[], tools=[tool]),
    )


def _after_context(result: AgentToolResult) -> AfterToolCallContext:
    tool = _tool("custom", read_only=True)
    return AfterToolCallContext(
        assistant_message=AssistantMessage(),
        tool_call=ToolCall(id="tc-after", name=tool.name, arguments={}),
        args={},
        result=result,
        is_error=False,
        context=AgentContext(system_prompt="", messages=[], tools=[tool]),
    )


class CodingAgentHookTests(unittest.TestCase):
    def test_workspace_path_guard_allows_inside_and_blocks_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = _tool(properties={"path": {"type": "string"}})
            guard = create_workspace_path_guard(tmp_dir)

            allowed = guard(_before_context(tool, {"path": "src/app.py"}), None)
            self.assertIsNone(allowed)

            outside = str(Path(tmp_dir).parent / "outside.txt")
            blocked = guard(_before_context(tool, {"path": outside}), None)
            self.assertIsNotNone(blocked)
            self.assertTrue(blocked.block if blocked else False)
            self.assertIn("workspace boundary", blocked.reason if blocked else "")

    def test_read_only_guard_blocks_tool_without_read_only_declaration(self) -> None:
        guard = create_read_only_tool_guard(True)
        blocked = guard(_before_context(_tool(read_only=False), {}), None)
        self.assertIsNotNone(blocked)
        self.assertTrue(blocked.block if blocked else False)

        allowed = guard(_before_context(_tool(read_only=True), {}), None)
        self.assertIsNone(allowed)

    def test_secret_redaction_hook_changes_content_and_details(self) -> None:
        api_key = "sk-live-" + "a" * 24
        bearer_token = "b" * 24
        result = AgentToolResult(
            content=[TextContent(text=f"api={api_key}; Authorization: Bearer {bearer_token}")],
            details={
                "api_key": api_key,
                "nested": [{"password": "correct horse battery staple"}],
                "token_count": 42,
            },
        )
        hook = create_secret_redaction_hook()

        changed = hook(_after_context(result), None)

        self.assertIsNotNone(changed)
        self.assertNotIn(api_key, changed.content[0].text if changed and changed.content else "")
        self.assertNotIn(bearer_token, changed.content[0].text if changed and changed.content else "")
        self.assertEqual(changed.details["api_key"], "[REDACTED]") if changed else None
        self.assertEqual(changed.details["nested"][0]["password"], "[REDACTED]") if changed else None
        self.assertEqual(changed.details["token_count"], 42) if changed else None

    def test_before_hook_exception_fails_closed_without_exposing_exception(self) -> None:
        def broken_hook(context, signal):
            raise RuntimeError("internal database password=do-not-show")

        composed = _compose_before_tool_call(broken_hook, [])
        result = asyncio.run(composed(_before_context(_tool(), {}), None))  # type: ignore[misc]

        self.assertIsNotNone(result)
        self.assertTrue(result.block if result else False)
        self.assertNotIn("database password", result.reason if result else "")

    def test_after_hook_exception_preserves_previous_result(self) -> None:
        original = AgentToolResult(content=[TextContent(text="original")], details={"ok": True})

        def broken_hook(context, signal):
            context.result.content = [TextContent(text="partial mutation")]
            raise RuntimeError("after hook failed")

        composed = _compose_after_tool_call(broken_hook, [])
        context = _after_context(original)
        result = asyncio.run(composed(context, None))  # type: ignore[misc]

        self.assertIsNone(result)
        self.assertEqual(context.result.content[0].text, "original")
        self.assertEqual(context.result.details, {"ok": True})

    def test_factory_installs_default_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                )
            )
            try:
                before = session.agent._options.before_tool_call
                after = session.agent._options.after_tool_call
                self.assertIsNotNone(before)
                self.assertIsNotNone(after)

                write_tool = next(tool for tool in session.agent.state.tools if tool.name == "write")
                outside = str(Path(tmp_dir).parent / "outside.txt")
                blocked = asyncio.run(
                    before(  # type: ignore[misc]
                        _before_context(write_tool, {"path": outside, "content": "x"}),
                        None,
                    )
                )
                self.assertTrue(blocked.block if blocked else False)

                result = AgentToolResult(content=[TextContent(text="key=" + "sk-live-" + "z" * 24)])
                changed = asyncio.run(after(_after_context(result), None))  # type: ignore[misc]
                self.assertIsNotNone(changed)
                self.assertNotIn("sk-live-" + "z" * 24, changed.content[0].text if changed else "")
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
