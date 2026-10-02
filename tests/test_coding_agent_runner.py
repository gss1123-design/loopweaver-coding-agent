from __future__ import annotations

import sys
import io
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coding_agent.extensions.types import RegisteredCommand, SkillSpec
from coding_agent.runner import RunOptions, _handle_interactive_command, run, run_interactive, run_rpc
from ai.types import AssistantMessage, TextContent


class _FakeSession:
    def __init__(self) -> None:
        self.session_id = "s1"
        self.messages = []
        self.last_trace = {
            "run_id": "r1",
            "status": "completed",
            "duration_ms": 120,
            "total_tokens": 10,
        }
        self._listeners = []
        self.extension_commands = {
            "ext_ping": type(
                "Cmd",
                (),
                {"name": "ext_ping", "description": "ext ping", "source": "extension", "handler": staticmethod(lambda ctx: "pong")},
            )()
        }

    async def prompt(self, text: str, *, images=None):
        _ = text, images
        for listener in list(self._listeners):
            listener({"type": "message_end", "message": {"role": "assistant"}})
        return []

    async def continue_run(self):
        for listener in list(self._listeners):
            listener({"type": "turn_end"})
        return []

    async def resume_run(self):
        self.messages.append(
            AssistantMessage(content=[TextContent(text="resumed")], stop_reason="stop")
        )
        return []

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def list_entry_ids(self):
        return ["e1", "e2"]

    def list_entries(self):
        return [
            {"id": "e1", "parent_id": None, "depth": 0, "is_leaf": False},
            {"id": "e2", "parent_id": "e1", "depth": 1, "is_leaf": True},
        ]

    def get_leaf_id(self):
        return "e2"

    def get_entry_path(self, entry_id: str):
        if entry_id == "e2":
            return ["e1", "e2"]
        return [entry_id]

    def get_session_tree(self):
        return [{"id": "e1", "children": [{"id": "e2", "children": []}]}]

    def fork_from_entry(self, entry_id: str):
        _ = entry_id
        forked = _FakeSession()
        forked.session_id = "forked_s"
        forked.close = lambda: None
        return forked

    def switch_to_entry(self, entry_id: str):
        _ = entry_id
        return None

    def pending_approvals(self):
        return []

    def approve_tool(self, tool_call_id: str):
        _ = tool_call_id
        return False

    def reject_tool(self, tool_call_id: str):
        _ = tool_call_id
        return False

    def close(self):
        return None


class CodingAgentRunnerDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_dispatch_print_mode_requires_prompt(self) -> None:
        session = _FakeSession()
        with self.assertRaises(ValueError):
            await run(RunOptions(mode="print", session=session, prompt=None))

    async def test_run_dispatch_rpc_mode(self) -> None:
        session = _FakeSession()
        with patch("coding_agent.runner.run_rpc", new_callable=AsyncMock) as rpc_mock:
            await run(RunOptions(mode="rpc", session=session))
        rpc_mock.assert_awaited_once()

    async def test_run_rpc_protocol(self) -> None:
        session = _FakeSession()
        outputs: list[str] = []
        stdin_data = "\n".join(
            [
                json.dumps({"id": "1", "type": "state"}),
                json.dumps({"id": "2", "type": "list_entries"}),
                json.dumps({"id": "3", "type": "show_tree"}),
                json.dumps({"id": "4", "type": "entry_path", "entry_id": "e2"}),
                json.dumps({"id": "5", "type": "fork_entry", "entry_id": "e1"}),
                json.dumps({"id": "6", "type": "switch_entry", "entry_id": "e2"}),
                json.dumps({"id": "7", "type": "get_commands"}),
                json.dumps({"id": "8", "type": "approvals"}),
                json.dumps({"id": "9", "type": "approve", "tool_call_id": "tc-missing"}),
                json.dumps({"id": "10", "type": "prompt", "text": "hello"}),
                json.dumps({"id": "11", "type": "continue"}),
                json.dumps({"id": "12", "type": "shutdown"}),
            ]
        )
        with patch("sys.stdin", io.StringIO(stdin_data)):
            await run_rpc(session, output=outputs.append)

        parsed = [json.loads(line) for line in outputs]
        self.assertEqual(parsed[0]["type"], "rpc_ready")
        commands = [item.get("command") for item in parsed if item.get("type") == "response"]
        self.assertIn("state", commands)
        self.assertIn("list_entries", commands)
        self.assertIn("show_tree", commands)
        self.assertIn("entry_path", commands)
        self.assertIn("fork_entry", commands)
        self.assertIn("switch_entry", commands)
        self.assertIn("get_commands", commands)
        self.assertIn("approvals", commands)
        self.assertIn("approve", commands)
        self.assertIn("prompt", commands)
        self.assertIn("continue", commands)
        self.assertIn("shutdown", commands)
        responses = [item for item in parsed if item.get("type") == "response"]
        self.assertTrue(all(item.get("status") == "ok" for item in responses))
        self.assertTrue(any(item.get("type") == "event" for item in parsed))
        get_cmd_resp = next(
            (item for item in responses if item.get("command") == "get_commands"),
            None,
        )
        self.assertIsNotNone(get_cmd_resp)
        cmd_names = [x.get("name") for x in get_cmd_resp.get("data", {}).get("commands", [])]
        self.assertIn("ext_ping", cmd_names)

    async def test_run_rpc_traces_returns_queryable_history(self) -> None:
        session = _FakeSession()
        outputs: list[str] = []
        stdin_data = "\n".join(
            [
                json.dumps({"id": "1", "type": "traces", "thread_id": "s1", "run_id": "r1"}),
                json.dumps({"id": "2", "type": "shutdown"}),
            ]
        )
        with patch("sys.stdin", io.StringIO(stdin_data)):
            await run_rpc(session, output=outputs.append)

        parsed = [json.loads(line) for line in outputs]
        response = next(item for item in parsed if item.get("command") == "traces")
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["data"]["thread_id"], "s1")
        self.assertEqual(response["data"]["traces"][0]["run_id"], "r1")

    async def test_run_interactive_session_commands(self) -> None:
        session = _FakeSession()
        outputs: list[str] = []
        inputs = iter(["/session", "/trace", "/tree", "/switch e2", "exit"])
        await run_interactive(
            session,
            input_fn=lambda _: next(inputs),
            output=outputs.append,
            show_tool_events=False,
        )
        joined = "\n".join(outputs)
        self.assertIn("session_id=s1", joined)
        self.assertIn('"duration_ms": 120', joined)
        self.assertIn("- e1", joined)
        self.assertIn("switched leaf", joined)
        self.assertIn("/ext_ping", joined)

    async def test_resume_command_uses_safe_session_resume(self) -> None:
        session = _FakeSession()
        resume_mock = AsyncMock(return_value=[])
        session.resume_run = resume_mock

        handled, switched = await _handle_interactive_command(
            session,
            "/resume",
            output=lambda _: None,
            show_tool_events=False,
        )

        self.assertTrue(handled)
        self.assertIsNone(switched)
        resume_mock.assert_awaited_once_with()

    async def test_skill_command_runs_a_real_prompt(self) -> None:
        session = _FakeSession()
        session.extension_commands["skill:review"] = RegisteredCommand(
            name="skill:review",
            description="review code",
            source="skill",
            handler=lambda ctx: "请执行代码审查",
        )
        prompt_mock = AsyncMock(return_value=[])
        session.prompt = prompt_mock

        handled, switched = await _handle_interactive_command(
            session,
            "/skill:review src/app.py",
            output=lambda _: None,
        )

        self.assertTrue(handled)
        self.assertIsNone(switched)
        prompt_mock.assert_awaited_once_with("请执行代码审查")

    async def test_skill_command_activates_skill_body_lazily(self) -> None:
        session = _FakeSession()
        skill = SkillSpec(
            name="代码审查",
            command_name="skill:review",
            description="review code",
            content="",
            source_path="review.md",
        )
        session.extension_commands["skill:review"] = RegisteredCommand(
            name="skill:review",
            description="review code",
            source="skill",
            handler=lambda ctx: "ignored compatibility text",
            skill=skill,
        )
        prompt_with_skill_mock = AsyncMock(return_value=[])
        session.prompt_with_skill = prompt_with_skill_mock

        handled, switched = await _handle_interactive_command(
            session,
            "/skill:review src/app.py",
            output=lambda _: None,
        )

        self.assertTrue(handled)
        self.assertIsNone(switched)
        prompt_with_skill_mock.assert_awaited_once_with(skill, "src/app.py")


if __name__ == "__main__":
    unittest.main()
