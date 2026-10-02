from __future__ import annotations

import asyncio
import inspect
import sys
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.types import AssistantMessage, TextContent
from coding_agent.approval import ApprovalGate
from coding_agent.memory import MemoryStore
from im.inbox import DurableInbox
from im.service import IMService, IMServiceConfig, _ChannelState
from im.types import IMIncomingMessage, IMOutgoingText, IMWebhookResult


class _FakeAdapter:
    def __init__(self, incoming: list[IMIncomingMessage]) -> None:
        self.incoming = incoming
        self.sent: list[IMOutgoingText] = []

    def handle_webhook(self, headers, body) -> IMWebhookResult:
        _ = headers, body
        return IMWebhookResult(ack={"code": 0}, messages=self.incoming)

    def send_text(self, message: IMOutgoingText) -> None:
        self.sent.append(message)


class _StreamingAdapter(_FakeAdapter):
    def __init__(self) -> None:
        super().__init__([])
        self.updated: list[tuple[str, str]] = []
        self.cards = []

    def send_card(self, message) -> str:
        self.cards.append(message)
        return "card-1"

    def update_text(self, message_id: str, text: str) -> None:
        self.updated.append((message_id, text))


class _FakeSession:
    def __init__(self) -> None:
        self.messages = []
        self.session_id = "im-test-session"
        self._leaf_id = "entry-2"
        self._entries = [
            {"id": "entry-1", "depth": 0, "is_leaf": False},
            {"id": "entry-2", "depth": 1, "is_leaf": True},
        ]
        self.last_trace = {
            "run_id": "run-test",
            "status": "completed",
            "total_tokens": 12,
        }
        self._listeners = []
        self.extension_commands = {
            "ext_ping": type(
                "Cmd",
                (),
                {"name": "ext_ping", "description": "ext ping", "source": "extension", "handler": staticmethod(lambda ctx: "pong")},
            )()
        }

    def query_traces(
        self,
        *,
        thread_id=None,
        run_id=None,
        operation_id=None,
        trace_id=None,
        limit=None,
    ):
        _ = operation_id, trace_id
        if thread_id not in (None, self.session_id):
            return []
        if run_id not in (None, self.last_trace["run_id"]):
            return []
        return [self.last_trace][:limit] if limit is not None else [self.last_trace]

    async def prompt(self, text: str) -> None:
        self.messages.append(
            AssistantMessage(
                content=[TextContent(text=f"收到：{text}")],
                stop_reason="stop",
            )
        )

    async def resume_run(self) -> None:
        self.messages.append(
            AssistantMessage(content=[TextContent(text="已恢复")], stop_reason="stop")
        )

    def close(self) -> None:
        return None

    def get_leaf_id(self):
        return self._leaf_id

    def list_entries(self):
        return list(self._entries)

    def subscribe(self, listener):
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None


class _SlowSession(_FakeSession):
    async def prompt(self, text: str) -> None:
        await asyncio.sleep(0.05)
        await super().prompt(text)


class _ApprovalSession(_FakeSession):
    def __init__(self, gate: ApprovalGate) -> None:
        super().__init__()
        self.approval_gate = gate

    def pending_approvals(self):
        return self.approval_gate.pending()

    def approve_tool(self, tool_call_id: str) -> bool:
        return self.approval_gate.approve(tool_call_id)

    def reject_tool(self, tool_call_id: str) -> bool:
        return self.approval_gate.reject(tool_call_id)

    def bind_tool_approval_requester(self, tool_call_id: str, requester_id: str) -> bool:
        return self.approval_gate.set_requester(tool_call_id, requester_id)

    def resolve_tool_approval(self, tool_call_id: str, *, approved: bool, actor_id: str | None = None) -> str:
        return self.approval_gate.resolve(tool_call_id, approved, actor_id=actor_id)


class _StreamingSession(_FakeSession):
    async def prompt(self, text: str, *, images=None):
        _ = text, images
        final = AssistantMessage(content=[TextContent(text="最终回答")], stop_reason="stop")
        for partial_text in ("最", "最终", "最终回", "最终回答"):
            event = {
                "type": "message_update",
                "message": AssistantMessage(
                    content=[TextContent(text=partial_text)],
                    stop_reason="stop",
                ),
            }
            for listener in list(self._listeners):
                value = listener(event)
                if inspect.isawaitable(value):
                    await value
            # Let the background flush task run between deltas.
            await asyncio.sleep(0)
        self.messages.append(final)
        return []


class IMServiceTests(unittest.TestCase):
    def test_durable_inbox_deduplicates_after_service_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = IMServiceConfig(
                workspace_dir=tmp_dir,
                provider="openai-standard",
                model_id="gpt-4o-mini",
            )
            message = IMIncomingMessage(
                platform="feishu",
                channel_id="c1",
                user_id="u1",
                text="hello",
                message_id="durable-1",
            )
            first_calls: list[str] = []

            async def first_run() -> None:
                service = IMService(adapter=_FakeAdapter([]), config=config)

                async def handle(current: IMIncomingMessage) -> None:
                    first_calls.append(str(current.message_id))

                service._handle_single_message = handle  # type: ignore[method-assign]
                await service.handle_incoming_message(message)
                service.close()

            asyncio.run(first_run())
            self.assertEqual(first_calls, ["durable-1"])

            second_calls: list[str] = []

            async def second_run() -> None:
                service = IMService(adapter=_FakeAdapter([]), config=config)

                async def handle(current: IMIncomingMessage) -> None:
                    second_calls.append(str(current.message_id))

                service._handle_single_message = handle  # type: ignore[method-assign]
                await service.handle_incoming_message(message)
                service.close()

            asyncio.run(second_run())
            self.assertEqual(second_calls, [])

    def test_durable_inbox_replays_pending_message_on_first_webhook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pending = IMIncomingMessage(
                platform="feishu",
                channel_id="c1",
                user_id="u1",
                text="replay me",
                message_id="pending-1",
            )
            DurableInbox(tmp_dir).accept(pending)
            current = IMIncomingMessage(
                platform="feishu",
                channel_id="c1",
                user_id="u1",
                text="new message",
                message_id="current-1",
            )
            calls: list[str] = []

            async def scenario() -> None:
                service = IMService(
                    adapter=_FakeAdapter([]),
                    config=IMServiceConfig(
                        workspace_dir=tmp_dir,
                        provider="openai-standard",
                        model_id="gpt-4o-mini",
                    ),
                )

                async def handle(message: IMIncomingMessage) -> None:
                    calls.append(str(message.message_id))

                service._handle_single_message = handle  # type: ignore[method-assign]
                await service.handle_incoming_message(current)
                service.close()

            asyncio.run(scenario())
            self.assertEqual(calls, ["pending-1", "current-1"])

    def test_handle_webhook_and_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="你好",
                        thread_id="t1",
                        message_id="m1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )

            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                ack = asyncio.run(service.handle_webhook({}, b"{}"))

            self.assertEqual(ack, {"code": 0})
            create_mock.assert_called_once()
            create_options = create_mock.call_args.args[0]
            self.assertEqual(create_options.max_context_messages, 24)
            self.assertEqual(create_options.max_context_tokens, 12000)
            self.assertEqual(create_options.retain_recent_messages, 8)
            self.assertEqual(create_options.max_tokens, 2048)
            self.assertIsNotNone(create_options.summary_builder)
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("收到：你好", adapter.sent[0].text)

    def test_router_reuses_same_session_for_same_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(platform="feishu", channel_id="c1", user_id="u1", text="a", thread_id="t1"),
                    IMIncomingMessage(platform="feishu", channel_id="c1", user_id="u1", text="b", thread_id="t1"),
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))

            # 同一频道/线程复用内存中的 AgentSession，只创建一次。
            self.assertEqual(create_mock.call_count, 1)

    def test_duplicate_message_id_is_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="你好",
                        thread_id="t1",
                        message_id="m-dup-1",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="你好",
                        thread_id="t1",
                        message_id="m-dup-1",
                    ),
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )

            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))

            # 同 message_id 只处理一次
            create_mock.assert_called_once()
            self.assertEqual(len(adapter.sent), 1)

    def test_session_command_does_not_call_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/session",
                        thread_id=None,
                        message_id="m-cmd-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))
            create_mock.assert_not_called()
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("当前会话", adapter.sent[0].text)

    def test_tree_and_trace_commands_do_not_call_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/tree",
                        thread_id=None,
                        message_id="m-tree-1",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/trace",
                        thread_id=None,
                        message_id="m-trace-1",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/traces",
                        thread_id=None,
                        message_id="m-traces-1",
                    ),
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            fake_session = _FakeSession()
            service._channel_states["feishu:c1:_"] = _ChannelState(
                session=fake_session,
                session_id=fake_session.session_id,
                last_active=time.time(),
                user_cache={},
            )

            with patch("im.service.create_agent_session") as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))

            create_mock.assert_not_called()
            self.assertEqual(len(adapter.sent), 3)
            self.assertIn("entry-2", adapter.sent[0].text)
            self.assertIn("run-test", adapter.sent[1].text)
            self.assertIn("traces=1", adapter.sent[2].text)
            self.assertIn("run_id=run-test", adapter.sent[2].text)
            self.assertEqual(fake_session.messages, [])

    def test_resume_command_uses_session_recovery_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/resume",
                        message_id="m-resume-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    stream_updates=False,
                    use_card_reply=False,
                ),
            )
            fake_session = _FakeSession()
            service._channel_states["feishu:c1:_"] = _ChannelState(
                session=fake_session,
                session_id=fake_session.session_id,
                last_active=time.time(),
                user_cache={},
            )

            asyncio.run(service.handle_webhook({}, b"{}"))

            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("已恢复", adapter.sent[0].text)

    def test_clear_command_rotates_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/session",
                        thread_id=None,
                        message_id="m-cmd-1",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/clear",
                        thread_id=None,
                        message_id="m-cmd-2",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/session",
                        thread_id=None,
                        message_id="m-cmd-3",
                    ),
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))
            create_mock.assert_not_called()
            self.assertEqual(len(adapter.sent), 3)
            first = adapter.sent[0].text
            third = adapter.sent[2].text
            self.assertIn("当前会话", first)
            self.assertIn("当前会话", third)
            self.assertNotEqual(first, third)

    def test_extension_slash_command_executes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/ext_ping a b",
                        thread_id=None,
                        message_id="m-ext-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            fake_session = _FakeSession()
            fake_session.extension_commands = {
                "ext_ping": type(
                    "Cmd",
                    (),
                    {"handler": staticmethod(lambda ctx: f"ok:{ctx.name}:{len(ctx.args)}")},
                )()
            }
            with patch("im.service.create_agent_session", return_value=fake_session) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))
            create_mock.assert_called_once()
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("ok:ext_ping:2", adapter.sent[0].text)

    def test_skill_slash_command_runs_agent_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/skill:review src/app.py",
                        thread_id=None,
                        message_id="m-skill-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    stream_updates=False,
                    use_card_reply=False,
                ),
            )
            fake_session = _FakeSession()
            fake_session.extension_commands = {
                "skill:review": type(
                    "Cmd",
                    (),
                    {
                        "name": "skill:review",
                        "source": "skill",
                        "handler": staticmethod(lambda ctx: "请执行代码审查"),
                    },
                )()
            }
            with patch("im.service.create_agent_session", return_value=fake_session):
                asyncio.run(service.handle_webhook({}, b"{}"))

            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("收到：请执行代码审查", adapter.sent[0].text)

    def test_help_command_includes_extension_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/help",
                        thread_id=None,
                        message_id="m-help-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    tool_approval=True,
                ),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()):
                asyncio.run(service.handle_webhook({}, b"{}"))
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("/ext_ping", adapter.sent[0].text)
            self.assertIn("/approve <tool_call_id>", adapter.sent[0].text)

    def test_channel_queue_limit_drops_excess_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter([])
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    channel_queue_limit=2,
                ),
            )
            messages = [
                IMIncomingMessage(
                    platform="feishu",
                    channel_id="c1",
                    user_id="u1",
                    text=f"m{i}",
                    thread_id=None,
                    message_id=f"m-{i}",
                )
                for i in range(5)
            ]
            with patch("im.service.create_agent_session", return_value=_SlowSession()) as create_mock:
                async def _run_all():
                    await asyncio.gather(*(service.handle_incoming_message(m) for m in messages))

                asyncio.run(_run_all())
            self.assertLess(create_mock.call_count, 5)

    def test_approval_command_bypasses_running_channel_queue(self) -> None:
        """审批命令必须直接唤醒 gate，不能排在等待审批的 prompt 后面。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter([])
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    tool_approval=True,
                ),
            )

            async def _scenario() -> None:
                gate = ApprovalGate(timeout_seconds=5)
                gate.begin({
                    "tool_call_id": "tc-1",
                    "tool_name": "run_tests",
                    "args": {"path": "."},
                })
                session = _ApprovalSession(gate)
                key = "feishu:c1:_"
                service._channel_states[key] = _ChannelState(
                    session=session,
                    session_id="s1",
                    last_active=time.time(),
                    user_cache={},
                    approval_gate=gate,
                )
                # 模拟该频道已经有一个 prompt 在等待工具审批。
                service._channel_running.add(key)
                service._channel_queues[key] = deque()

                await service.handle_incoming_message(
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/approve tc-1",
                        message_id="approval-1",
                    )
                )

                self.assertEqual(gate.pending(), [])
                self.assertEqual(session.messages, [])
                self.assertEqual(len(adapter.sent), 1)
                self.assertIn("已允许", adapter.sent[0].text)

            asyncio.run(_scenario())

    def test_approval_notice_binds_requester_and_redacts_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter([])
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    tool_approval=True,
                ),
            )

            async def _scenario() -> None:
                gate = ApprovalGate(timeout_seconds=5)
                gate.begin(
                    {
                        "tool_call_id": "tc-secret",
                        "tool_name": "bash",
                        "args": {
                            "command": "API_KEY=very-secret-value python task.py",
                            "token": "another-secret",
                            "content": "x" * 500,
                        },
                    }
                )
                session = _ApprovalSession(gate)
                service._channel_states["feishu:c1:_"] = _ChannelState(
                    session=session,
                    session_id="s1",
                    last_active=time.time(),
                    user_cache={},
                    approval_gate=gate,
                )
                message = IMIncomingMessage(
                    platform="feishu",
                    channel_id="c1",
                    user_id="request-owner",
                    text="run it",
                    message_id="message-1",
                )
                await service._send_approval_notice(
                    message,
                    {
                        "type": "approval_required",
                        "toolCallId": "tc-secret",
                        "toolName": "bash",
                        "args": gate.pending()[0]["args"],
                    },
                )

                pending = gate.pending()[0]
                self.assertEqual(pending["requester_id"], "request-owner")
                notice = adapter.sent[0].text
                self.assertNotIn("very-secret-value", notice)
                self.assertNotIn("another-secret", notice)
                self.assertNotIn("x" * 200, notice)
                self.assertIn("<redacted>", notice)
                self.assertIn("characters omitted", notice)

            asyncio.run(_scenario())

    def test_approval_notice_uses_clickable_card_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _StreamingAdapter()
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    tool_approval=True,
                    use_card_reply=True,
                ),
            )
            gate = ApprovalGate(timeout_seconds=30)
            fake_session = _ApprovalSession(gate)
            service._channel_states["feishu:c1:_"] = _ChannelState(
                session=fake_session,
                session_id=fake_session.session_id,
                last_active=time.time(),
                user_cache={},
                approval_gate=gate,
            )
            message = IMIncomingMessage(
                platform="feishu",
                channel_id="c1",
                user_id="owner",
                text="task",
                message_id="m-card",
            )

            async def _scenario() -> None:
                gate.begin({"tool_call_id": "tc-card", "tool_name": "write", "args": {}})
                await service._send_approval_notice(
                    message,
                    {
                        "type": "approval_required",
                        "toolCallId": "tc-card",
                        "toolName": "write",
                        "args": {"path": "a.txt"},
                    },
                )

            asyncio.run(_scenario())

            self.assertEqual(len(adapter.cards), 1)
            buttons = adapter.cards[0].buttons
            self.assertEqual(
                [button.value["decision"] for button in buttons],
                ["approve", "reject"],
            )
            self.assertTrue(
                all(button.value["tool_call_id"] == "tc-card" for button in buttons)
            )
            self.assertEqual(gate.pending()[0]["requester_id"], "owner")

    def test_only_request_owner_can_approve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter([])
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    tool_approval=True,
                ),
            )

            async def _scenario() -> None:
                gate = ApprovalGate(timeout_seconds=5)
                gate.begin({"tool_call_id": "tc-owner", "tool_name": "write", "args": {}})
                gate.set_requester("tc-owner", "owner")
                session = _ApprovalSession(gate)
                service._channel_states["feishu:c1:_"] = _ChannelState(
                    session=session,
                    session_id="s1",
                    last_active=time.time(),
                    user_cache={},
                    approval_gate=gate,
                )

                await service.handle_incoming_message(
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="other-user",
                        text="/approve tc-owner",
                        message_id="approval-other",
                    )
                )

                self.assertEqual(len(gate.pending()), 1)
                self.assertIn("不是这次任务的发起者", adapter.sent[-1].text)

            asyncio.run(_scenario())

    def test_streaming_keeps_only_latest_preview_and_skips_duplicate_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _StreamingAdapter()
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    show_cost_in_reply=False,
                ),
            )
            message = IMIncomingMessage(
                platform="feishu",
                channel_id="c1",
                user_id="u1",
                text="hello",
                message_id="m-stream-1",
            )

            async def _run() -> str:
                with patch("im.service._STREAM_UPDATE_INTERVAL", 0.01):
                    return await service._prompt_with_streaming_fast(
                        _StreamingSession(), "hello", message
                    )

            result = asyncio.run(_run())
            self.assertEqual(result, "最终回答")
            self.assertLessEqual(len(adapter.updated), 2)
            self.assertEqual(adapter.updated[-1][1], "最终回答")
            self.assertEqual(
                len([text for _, text in adapter.updated if text == "最终回答"]),
                1,
            )

    def test_stale_message_is_queued_when_channel_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = IMService(
                adapter=_FakeAdapter([]),
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    stale_event_seconds=1,
                ),
            )
            key = "feishu:c1:_"
            service._channel_running.add(key)
            message = IMIncomingMessage(
                platform="feishu",
                channel_id="c1",
                user_id="u1",
                text="排队的问题",
                message_id="m-stale-busy",
                created_at=time.time() - 60,
            )

            asyncio.run(service.handle_incoming_message(message))

            self.assertEqual(len(service._channel_queues[key]), 1)
            self.assertNotIn("m-stale-busy", service._processed_ids)


class TestConfirmedMemoryCommands(unittest.TestCase):
    def test_group_chat_cannot_start_memory_flow(self) -> None:
        async def scenario(workspace: Path) -> None:
            adapter = _FakeAdapter([])
            service = IMService(adapter, IMServiceConfig(
                workspace_dir=workspace, provider="fixture", model_id="fixture", enable_structured_memory=True,
            ))
            try:
                await service.handle_incoming_message(IMIncomingMessage(
                    platform="feishu", channel_id="group1", user_id="u1",
                    text="/remember fact region", message_id="g1",
                    raw={"event": {"message": {"chat_type": "group"}}},
                ))
                self.assertIn("仅支持与机器人私聊", adapter.sent[-1].text)
                self.assertFalse(service._pending_memories)
            finally:
                service.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            asyncio.run(scenario(Path(temp_dir)))

    def test_confirm_forget_and_duplicate_delivery(self) -> None:
        async def scenario(workspace: Path) -> None:
            adapter = _FakeAdapter([])
            service = IMService(adapter, IMServiceConfig(
                workspace_dir=workspace, provider="fixture", model_id="fixture",
                enable_structured_memory=True,
            ))

            async def send(text: str, message_id: str) -> None:
                await service.handle_incoming_message(IMIncomingMessage(
                    platform="feishu", channel_id="c1", user_id="u1",
                    text=text, message_id=message_id,
                ))

            try:
                await send("记住 preference editor", "m1")
                await send("我喜欢 Vim", "m2")
                self.assertIsNone(MemoryStore(workspace).get("im:feishu:c1", "editor"))
                await send("确认记忆", "m3")
                row = MemoryStore(workspace).get("im:feishu:c1", "editor")
                self.assertEqual(row["value"], "我喜欢 Vim")
                self.assertEqual(MemoryStore(workspace).search("im:feishu:c1", "Vim")[0]["source_check"], "matched_user_entry")
                await send("确认记忆", "m3")
                self.assertEqual(len([m for m in adapter.sent if "已保存记忆" in m.text]), 1)
                await send("忘记 editor", "m4")
                self.assertIsNone(MemoryStore(workspace).get("im:feishu:c1", "editor"))
                await send("记住 fact region", "m5")
                await send("部署在新加坡", "m6")
                await send("取消记忆", "m7")
                self.assertIsNone(MemoryStore(workspace).get("im:feishu:c1", "region"))
            finally:
                service.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            asyncio.run(scenario(Path(temp_dir)))

    def test_unconfirmed_quote_does_not_survive_restart(self) -> None:
        async def scenario(workspace: Path) -> None:
            adapter = _FakeAdapter([])
            config = IMServiceConfig(workspace_dir=workspace, provider="fixture", model_id="fixture", enable_structured_memory=True)
            first = IMService(adapter, config)
            await first.handle_incoming_message(IMIncomingMessage(
                platform="feishu", channel_id="c1", user_id="u1", text="/remember fact region", message_id="m1",
            ))
            await first.handle_incoming_message(IMIncomingMessage(
                platform="feishu", channel_id="c1", user_id="u1", text="部署在新加坡", message_id="m2",
            ))
            first.close()
            second = IMService(adapter, config)
            try:
                await second.handle_incoming_message(IMIncomingMessage(
                    platform="feishu", channel_id="c1", user_id="u1", text="/confirm", message_id="m3",
                ))
                self.assertIn("没有待确认", adapter.sent[-1].text)
                self.assertIsNone(MemoryStore(workspace).get("im:feishu:c1", "region"))
            finally:
                second.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            asyncio.run(scenario(Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
