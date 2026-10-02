from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.models import get_model
from ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from coding_agent.agent_session import AgentSession
from coding_agent.factory import create_agent_session
from coding_agent.serde import message_from_dict, message_to_dict
from coding_agent.session_store import SessionStore
from coding_agent.types import AgentSessionOptions, CreateAgentSessionOptions


class CodingAgentStoreTests(unittest.TestCase):
    def test_resume_dispatches_from_user_or_tool_result_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                )
            )
            session.agent.set_messages([UserMessage(content="unfinished")])
            continued = AsyncMock(return_value=[])

            with patch.object(session, "continue_run", new=continued):
                asyncio.run(session.resume_run())

            continued.assert_awaited_once_with()
            session.close()

    def test_resume_length_adds_explicit_continuation_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                )
            )
            session.agent.set_messages(
                [AssistantMessage(content=[TextContent(text="partial")], stop_reason="length")]
            )
            prompted = AsyncMock(return_value=[])

            with patch.object(session, "prompt", new=prompted):
                asyncio.run(session.resume_run())

            continuation_text = prompted.await_args.args[0]
            self.assertIn("输出长度限制", continuation_text)
            self.assertIn("不要重复", continuation_text)
            session.close()

    def test_resume_refuses_uncertain_tool_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                )
            )
            session.agent.set_messages(
                [
                    AssistantMessage(
                        content=[ToolCall(id="tc1", name="write", arguments={})],
                        stop_reason="toolUse",
                    )
                ]
            )

            with self.assertRaisesRegex(ValueError, "无法确认工具是否已经产生副作用"):
                asyncio.run(session.resume_run())
            session.close()

    def test_resume_retries_failed_assistant_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                    retry_enabled=False,
                )
            )
            session.agent.set_messages(
                [AssistantMessage(content=[], stop_reason="error", error_message="timeout")]
            )

            async def recovered(**_kwargs):
                final = AssistantMessage(content=[TextContent(text="ok")], stop_reason="stop")
                session.agent.set_messages([final])
                return [final]

            retry = AsyncMock(side_effect=recovered)
            with patch.object(session.agent, "retry_last_run", new=retry):
                result = asyncio.run(session.resume_run())

            self.assertEqual(result[0].stop_reason, "stop")
            retry.assert_awaited_once()
            session.close()

    def test_session_store_operation_journal_is_durable_and_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="journal")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")

            store.start_operation("op-open", "agent_run", {"attempt": 1})
            store.append_operation_event("op-open", "tool_started", {"tool": "read"})
            store.start_operation("op-done", "agent_run")
            store.finish_operation("op-done", status="succeeded")

            # 模拟进程在最后一条 JSON 尚未写完时崩溃；完整记录仍可读取。
            with store.journal_file.open("a", encoding="utf-8") as fp:
                fp.write('{"seq": 999, "kind": "operation_started"')

            journal = store.load_journal()
            self.assertEqual([entry["seq"] for entry in journal], [1, 2, 3, 4])
            self.assertEqual(
                [entry["operation_id"] for entry in store.load_incomplete_operations()],
                ["op-open"],
            )
            self.assertEqual(journal[0]["payload"]["operation"], "agent_run")

    def test_agent_session_journals_run_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                )
            )

            async def successful_op() -> list:
                return []

            asyncio.run(session._run_with_retry(successful_op))
            entries = session.store.load_journal()
            self.assertEqual([entry["kind"] for entry in entries], ["operation_started", "operation_finished"])
            self.assertEqual(entries[-1]["payload"]["status"], "succeeded")
            self.assertEqual(session.store.load_incomplete_operations(), [])
            session.close()

    def test_validate_message_sequence_accepts_complete_tool_batch(self) -> None:
        messages = [
            UserMessage(content="run tools"),
            AssistantMessage(
                content=[
                    ToolCall(id="tc1", name="read", arguments={}),
                    ToolCall(id="tc2", name="grep", arguments={}),
                ],
                stop_reason="toolUse",
            ),
            ToolResultMessage(tool_call_id="tc2", tool_name="grep", content=[TextContent(text="ok")]),
            ToolResultMessage(tool_call_id="tc1", tool_name="read", content=[TextContent(text="ok")]),
            AssistantMessage(content=[TextContent(text="done")], stop_reason="stop"),
        ]

        self.assertEqual(AgentSession._validate_message_sequence(messages), [])

    def test_validate_message_sequence_reports_missing_orphan_duplicate_and_mismatch(self) -> None:
        messages = [
            UserMessage(content="run tools"),
            AssistantMessage(
                content=[
                    ToolCall(id="tc-missing", name="read", arguments={}),
                    ToolCall(id="tc-duplicate", name="write", arguments={}),
                    ToolCall(id="tc-duplicate", name="write", arguments={}),
                ],
                stop_reason="toolUse",
            ),
            ToolResultMessage(
                tool_call_id="tc-duplicate",
                tool_name="read",
                content=[TextContent(text="wrong owner")],
            ),
            ToolResultMessage(
                tool_call_id="tc-duplicate",
                tool_name="write",
                content=[TextContent(text="duplicate")],
            ),
            ToolResultMessage(
                tool_call_id="tc-orphan",
                tool_name="read",
                content=[TextContent(text="orphan")],
            ),
        ]

        errors = AgentSession._validate_message_sequence(messages)
        self.assertTrue(any("missing ToolResult" in error for error in errors))
        self.assertTrue(any("duplicate ToolCall" in error for error in errors))
        self.assertTrue(any("duplicate ToolResult" in error for error in errors))
        self.assertTrue(any("orphan ToolResult" in error for error in errors))
        self.assertTrue(any("tool name mismatch" in error for error in errors))

    def test_compaction_persists_integrity_warning_without_mutating_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                    max_context_messages=3,
                    retain_recent_messages=1,
                    summary_builder=lambda _messages: "summary",
                )
            )
            invalid_messages = [
                UserMessage(content="run it"),
                AssistantMessage(
                    content=[ToolCall(id="tc-missing", name="bash", arguments={})],
                    stop_reason="toolUse",
                ),
                UserMessage(content="continue"),
                UserMessage(content="latest"),
            ]
            session.agent.set_messages(invalid_messages)

            asyncio.run(session._compact_context_if_needed())

            self.assertTrue(
                any(
                    line.get("type") == "tool_integrity_warning"
                    for line in (
                        json.loads(raw)
                        for raw in session.store.events_file.read_text(encoding="utf-8").splitlines()
                        if raw.strip()
                    )
                )
            )
            # 验证阶段只告警，不擅自删除或补写原消息；压缩仍按既有策略完成。
            self.assertTrue(session.agent.state.messages)
            self.assertTrue(
                any(
                    isinstance(message, UserMessage)
                    and isinstance(message.content, list)
                    and any(
                        isinstance(block, TextContent) and block.text.startswith("[Context Summary]")
                        for block in message.content
                    )
                    for message in session.agent.state.messages
                )
            )
            session.close()

    def test_message_roundtrip(self) -> None:
        messages = [
            UserMessage(content=[TextContent(text="hello")], timestamp=1),
            AssistantMessage(
                content=[TextContent(text="ok"), ToolCall(id="tc1", name="tool_a", arguments={"x": 1})],
                stop_reason="toolUse",
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="tc1",
                tool_name="tool_a",
                content=[TextContent(text="done")],
                is_error=False,
                timestamp=3,
            ),
        ]

        rebuilt = [message_from_dict(message_to_dict(m)) for m in messages]
        self.assertEqual(len(rebuilt), 3)
        self.assertEqual(getattr(rebuilt[0], "role", ""), "user")
        self.assertEqual(getattr(rebuilt[1], "role", ""), "assistant")
        self.assertEqual(getattr(rebuilt[2], "role", ""), "toolResult")

    def test_session_store_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")

            store.append_context_message(UserMessage(content="hello"))
            store.append_event({"type": "agent_start", "runId": "r1"})

            loaded = store.load_context_messages()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(getattr(loaded[0], "role", ""), "user")
            loaded_from_session = store.load_session_messages()
            self.assertEqual(len(loaded_from_session), 1)

            meta = json.loads((Path(tmp_dir) / ".xingclaw" / "sessions" / "s1" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["session_id"], "s1")

    def test_factory_resolve_from_provider_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="anthropic",
                    model_id="claude-sonnet-4-5",
                    system_prompt="test",
                )
            )
            self.assertEqual(session.agent.state.model.id, "claude-sonnet-4-5")
            session.close()

    def test_factory_restore_from_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    system_prompt="restored-system",
                )
            )
            sid = first.session_id
            first.close()

            restored = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    session_id=sid,
                )
            )
            self.assertEqual(restored.agent.state.model.id, "gpt-4o-mini")
            self.assertEqual(restored.agent.state.system_prompt, "restored-system")
            restored.close()

    def test_split_keeps_tool_call_chain_intact(self) -> None:
        """压缩边界不能把 ToolCall 和它的 ToolResult 拆到两边。

        拆开会让发给 provider 的历史出现无主 ToolResult（或无结果 ToolCall），
        DeepSeek 等会直接返回 400。
        """
        messages = [
            UserMessage(content="u1"),
            AssistantMessage(content=[TextContent(text="a1")]),
            UserMessage(content="u2"),
            AssistantMessage(
                content=[ToolCall(id="call_1", name="read_file", arguments={})]
            ),
            ToolResultMessage(tool_call_id="call_1", content="file body"),
            AssistantMessage(content=[TextContent(text="done")]),
        ]

        # retain=2 的朴素切片会落在 ToolCall(索引3) 与 ToolResult(索引4) 之间。
        older, recent = AgentSession._split_for_compaction(messages, retain=2)

        self.assertTrue(older, "older 不能为空，否则没有内容可摘要")
        self.assertNotIsInstance(
            recent[0], ToolResultMessage, "recent 不能以孤立 ToolResult 开头"
        )

        older_call_ids = {
            block.id
            for message in older
            if isinstance(message, AssistantMessage)
            for block in message.content
            if isinstance(block, ToolCall)
        }
        recent_result_ids = {
            message.tool_call_id
            for message in recent
            if isinstance(message, ToolResultMessage)
        }
        self.assertFalse(
            older_call_ids & recent_result_ids,
            "ToolCall 留在 older，对应 ToolResult 却进了 recent",
        )
        self.assertEqual(len(older) + len(recent), len(messages))

    def test_context_compaction_rewrites_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                    max_context_messages=4,
                    retain_recent_messages=2,
                    summary_builder=lambda _messages: "summary",
                )
            )

            session.agent.set_messages(
                [
                    UserMessage(content="u1"),
                    AssistantMessage(content=[TextContent(text="a1")]),
                    UserMessage(content="u2"),
                    AssistantMessage(content=[TextContent(text="a2")]),
                    UserMessage(content="u3"),
                    AssistantMessage(content=[TextContent(text="a3")]),
                ]
            )
            asyncio.run(session._compact_context_if_needed())  # 测试私有策略入口

            compacted = session.agent.state.messages
            self.assertEqual(len(compacted), 3)
            self.assertEqual(getattr(compacted[0], "role", ""), "user")

            summary_text = ""
            first = compacted[0]
            if isinstance(first, UserMessage) and isinstance(first.content, list):
                summary_text = "".join(b.text for b in first.content if isinstance(b, TextContent))
            self.assertIn("[Context Summary]", summary_text)

            reloaded = session.store.load_context_messages()
            self.assertEqual(len(reloaded), 3)
            session.close()

    def test_context_compaction_happens_before_next_prompt(self) -> None:
        """达到消息阈值时，不能等本次请求发完才压缩。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                    max_context_messages=4,
                    retain_recent_messages=24,
                    summary_builder=lambda _messages: "summary",
                )
            )
            session.agent.set_messages(
                [
                    UserMessage(content="u1"),
                    AssistantMessage(content=[TextContent(text="a1")]),
                    UserMessage(content="u2"),
                    AssistantMessage(content=[TextContent(text="a2")]),
                    UserMessage(content="u3"),
                    AssistantMessage(content=[TextContent(text="a3")]),
                ]
            )

            asyncio.run(session._check_and_compact_before_prompt())

            # summary + 最近 3 条 = 4 条，下一次请求不会继续超出消息阈值。
            self.assertEqual(len(session.agent.state.messages), 4)
            session.close()

    def test_session_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.append_context_message(UserMessage(content="u1"))
            store.append_context_message(AssistantMessage(content=[TextContent(text="a1")]))

            entry_ids = store.list_entry_ids()
            self.assertGreaterEqual(len(entry_ids), 2)
            forked = store.fork_to("s2", from_entry_id=entry_ids[0])
            loaded = forked.load_session_messages()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(getattr(loaded[0], "role", ""), "user")

    def test_agent_session_lane_is_persistent_and_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = AgentSessionOptions(
                model=get_model("openai-standard", "gpt-4o-mini"),
                workspace_dir=tmp_dir,
            )
            session = AgentSession(options)
            session.store.append_context_message(UserMessage(content="parent question"))
            session.store.append_context_message(
                AssistantMessage(content=[TextContent(text="parent context")])
            )
            parent_entry_ids = session.store.list_entry_ids()

            lane = session.create_lane("research", from_entry_id=parent_entry_ids[-1])
            lane_id = session.list_lanes()[0]["lane_id"]
            lane_info = session.list_lanes()[0]
            self.assertEqual(lane_info["name"], "research")
            self.assertEqual(lane_info["status"], "active")
            self.assertNotEqual(lane.session_id, session.session_id)
            self.assertEqual([message.role for message in lane.messages], ["user", "assistant"])
            self.assertEqual(session.store.list_entry_ids(), parent_entry_ids)

            # The parent journal is enough to reopen a lane after a process
            # restart; the restored lane has its own session tree/messages.
            reopened = AgentSession(
                replace(options, session_id=session.session_id)
            )
            restored = reopened.get_lane(lane_id)
            self.assertEqual(restored.session_id, lane.session_id)
            self.assertEqual([message.role for message in restored.messages], ["user", "assistant"])

            reopened.close_lane(lane_id)
            self.assertEqual(reopened.list_lanes()[0]["status"], "closed")
            with self.assertRaises(ValueError):
                reopened.get_lane(lane_id)
            reopened.close()
            session.close()

    def test_subagent_tool_is_registered_and_lane_cannot_recurse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                )
            )
            parent_names = {tool.name for tool in session.agent.state.tools}
            self.assertIn("run_subagent", parent_names)

            lane = session.create_lane("research", read_only=True)
            child_names = {tool.name for tool in lane.agent.state.tools}
            self.assertNotIn("run_subagent", child_names)
            self.assertTrue(session.list_lanes()[0]["read_only"])
            lane_id = session.list_lanes()[0]["lane_id"]
            session.close_lane(lane_id)
            session.close()

    def test_subagent_tool_returns_final_answer_and_closes_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = AgentSession(
                AgentSessionOptions(
                    model=get_model("openai-standard", "gpt-4o-mini"),
                    workspace_dir=tmp_dir,
                    enable_subagent_tool=False,
                )
            )

            class FakeLane:
                session_id = "s1_lane_lane_fake"
                last_trace = {"trace_id": "lane-op", "status": "completed"}

                def __init__(self) -> None:
                    self.task = ""

                async def prompt(self, task: str, *, extra_system_prompt=None) -> list:
                    self.task = task
                    return [AssistantMessage(content=[TextContent(text="子 agent 的结论")], stop_reason="stop")]

            fake_lane = FakeLane()
            session._lanes["lane_fake"] = fake_lane  # type: ignore[assignment]
            created: list[dict] = []
            closed: list[str] = []

            def fake_create_lane(name: str, *, read_only: bool = False):
                created.append({"name": name, "read_only": read_only})
                return fake_lane

            def fake_close_lane(lane_id: str) -> None:
                closed.append(lane_id)
                session._lanes.pop(lane_id, None)

            session.create_lane = fake_create_lane  # type: ignore[method-assign]
            session.close_lane = fake_close_lane  # type: ignore[method-assign]
            tool = session._build_subagent_tool()
            result = asyncio.run(tool.execute("tc1", {"task": "独立检查认证流程", "lane_name": "review"}))

            self.assertEqual(fake_lane.task, "独立检查认证流程")
            self.assertEqual(result.content[0].text, "子 agent 的结论")
            self.assertEqual(result.details["lane_id"], "lane_fake")
            self.assertEqual(created, [{"name": "review", "read_only": True}])
            self.assertEqual(closed, ["lane_fake"])
            session.close()

    def test_compaction_entry_preserves_old_tree_and_restores_short_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            original = [
                UserMessage(content="u1"),
                AssistantMessage(content=[TextContent(text="a1")]),
                UserMessage(content="u2"),
                AssistantMessage(content=[TextContent(text="a2")]),
            ]
            for message in original:
                store.append_context_message(message)
            old_ids = store.list_entry_ids()

            summary = UserMessage(content=[TextContent(text="[Context Summary]\nold facts")])
            new_ids = store.append_compaction_entry(
                summary,
                original[-2:],
                metadata={"reason": "threshold"},
            )

            all_ids = store.list_entry_ids()
            self.assertTrue(set(old_ids).issubset(all_ids))
            self.assertEqual(store.get_leaf_id(), new_ids[-1])
            self.assertEqual(
                [message.role for message in store.load_session_messages()],
                ["user", "user", "assistant"],
            )
            self.assertEqual(
                store.load_session_messages()[0].content[0].text,
                "[Context Summary]\nold facts",
            )

            # 切回压缩前的旧 leaf 仍可得到完整原始历史。
            store.set_leaf(old_ids[-1])
            self.assertEqual(len(store.load_session_messages()), len(original))
            self.assertEqual(store.get_entry_path(new_ids[-1])[: len(old_ids)], old_ids)

    def test_session_tree_and_switch_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionStore(workspace_dir=tmp_dir, session_id="s1")
            store.ensure_initialized(model_id="m1", provider="p1", system_prompt="sys")
            store.append_context_message(UserMessage(content="u1"))
            store.append_context_message(AssistantMessage(content=[TextContent(text="a1")]))
            store.append_context_message(UserMessage(content="u2"))

            tree = store.get_session_tree()
            self.assertEqual(len(tree), 1)
            self.assertEqual(tree[0]["role"], "user")
            self.assertEqual(len(tree[0]["children"]), 1)

            ids = store.list_entry_ids()
            self.assertGreaterEqual(len(ids), 3)
            path = store.get_entry_path(ids[2])
            self.assertEqual(path, [ids[0], ids[1], ids[2]])
            store.set_leaf(ids[1])
            self.assertEqual(store.get_leaf_id(), ids[1])
            branch = store.load_session_messages()
            self.assertEqual(len(branch), 2)
            entries = store.list_entries()
            self.assertEqual(len(entries), 3)
            leaf_entries = [item for item in entries if item.get("is_leaf")]
            self.assertEqual(len(leaf_entries), 1)
            self.assertEqual(leaf_entries[0]["id"], ids[1])


if __name__ == "__main__":
    unittest.main()
