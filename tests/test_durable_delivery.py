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

from coding_agent.approval import ApprovalGate
from im.inbox import DurableInbox
from im.types import IMIncomingMessage


class DurableDeliveryTests(unittest.TestCase):
    def test_approval_gate_persists_and_restores_pending_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "approval.json"

            async def begin_request() -> None:
                gate = ApprovalGate(timeout_seconds=5, state_path=state_path)
                gate.begin({"tool_call_id": "tc-1", "tool_name": "bash", "args": {"command": "echo ok"}})
                self.assertEqual(gate.pending()[0]["tool_call_id"], "tc-1")

            asyncio.run(begin_request())
            restored = ApprovalGate(timeout_seconds=5, state_path=state_path)
            self.assertEqual(restored.pending()[0]["tool_name"], "bash")
            self.assertTrue(restored.pending()[0]["recovered"])
            # The original event loop and suspended tool task no longer
            # exist, so a restart must not report a fake approval success.
            self.assertEqual(restored.resolve("tc-1", True), "stale")
            self.assertEqual(restored.pending(), [])

            reloaded = ApprovalGate(timeout_seconds=5, state_path=state_path)
            self.assertEqual(reloaded.pending(), [])

    def test_approval_gate_enforces_requester_identity(self) -> None:
        async def scenario() -> None:
            gate = ApprovalGate(timeout_seconds=5)
            gate.begin({"tool_call_id": "tc-owner", "tool_name": "write", "args": {}})
            self.assertTrue(gate.set_requester("tc-owner", "user-owner"))
            self.assertEqual(
                gate.resolve("tc-owner", True, actor_id="user-other"),
                "unauthorized",
            )
            self.assertEqual(len(gate.pending()), 1)

            waiting = asyncio.create_task(gate.wait("tc-owner"))
            await asyncio.sleep(0)
            self.assertEqual(
                gate.resolve("tc-owner", True, actor_id="user-owner"),
                "approved",
            )
            self.assertTrue(await waiting)
            self.assertEqual(gate.pending(), [])

        asyncio.run(scenario())

    def test_durable_inbox_deduplicates_and_recovers_processing_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            message = IMIncomingMessage(
                platform="feishu",
                channel_id="chat-1",
                user_id="user-1",
                text="hello",
                thread_id="thread-1",
                message_id="msg-1",
                created_at=123.0,
                raw={"event": "message"},
            )
            inbox = DurableInbox(tmp_dir)
            self.assertTrue(inbox.accept(message))
            self.assertFalse(inbox.accept(message))
            inbox.mark_processing(message.message_id)

            # 新实例代表 worker 重启；processing 消息必须回到待处理列表。
            restarted = DurableInbox(tmp_dir)
            pending = restarted.pending_messages()
            self.assertEqual([item.message_id for item in pending], ["msg-1"])
            self.assertEqual(pending[0].raw, {"event": "message"})

            restarted.mark_completed("msg-1")
            self.assertEqual(DurableInbox(tmp_dir).pending_messages(), [])
            self.assertEqual(restarted.status("msg-1"), "completed")

            # 尾部半行来自崩溃时仍应可读。
            with restarted.path.open("a", encoding="utf-8") as fp:
                fp.write('{"seq": 99, "message_id": "broken"')
            self.assertEqual(DurableInbox(tmp_dir).status("msg-1"), "completed")


if __name__ == "__main__":
    unittest.main()
