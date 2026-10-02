from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from im.feishu import FeishuAdapter, FeishuAdapterConfig
from im.types import IMCardButton


class FeishuAdapterTests(unittest.TestCase):
    def test_approval_card_contains_interactive_buttons(self) -> None:
        content = FeishuAdapter._build_card_content(
            "审批",
            "是否执行？",
            buttons=[
                IMCardButton(
                    text="允许",
                    style="primary",
                    value={
                        "xingclaw_action": "tool_approval",
                        "decision": "approve",
                        "tool_call_id": "tc-1",
                        "thread_id": "omt_1",
                    },
                )
            ],
        )
        card = json.loads(content)
        action = card["elements"][1]["actions"][0]
        self.assertEqual(action["tag"], "button")
        self.assertEqual(action["type"], "primary")
        self.assertEqual(action["value"]["tool_call_id"], "tc-1")

    def test_parse_http_approval_card_callback(self) -> None:
        adapter = FeishuAdapter(FeishuAdapterConfig(app_id="a", app_secret="b"))
        payload = {
            "open_id": "ou_owner",
            "open_chat_id": "oc_1",
            "open_message_id": "om_card",
            "action": {
                "value": {
                    "xingclaw_action": "tool_approval",
                    "decision": "approve",
                    "tool_call_id": "tc-1",
                    "thread_id": "omt_1",
                }
            },
        }

        result = adapter.handle_webhook({}, json.dumps(payload).encode("utf-8"))

        self.assertEqual(result.messages[0].text, "/approve tc-1")
        self.assertEqual(result.messages[0].user_id, "ou_owner")
        self.assertEqual(result.messages[0].channel_id, "oc_1")
        self.assertEqual(result.messages[0].thread_id, "omt_1")
        self.assertIn("toast", result.ack)

    def test_url_verification(self) -> None:
        adapter = FeishuAdapter(FeishuAdapterConfig(app_id="a", app_secret="b"))
        payload = {"type": "url_verification", "challenge": "abc"}
        result = adapter.handle_webhook({}, json.dumps(payload).encode("utf-8"))
        self.assertEqual(result.ack.get("challenge"), "abc")
        self.assertEqual(len(result.messages), 0)

    def test_parse_text_message(self) -> None:
        adapter = FeishuAdapter(FeishuAdapterConfig(app_id="a", app_secret="b", verify_token="vt"))
        payload = {
            "token": "vt",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_id": "oc_x1",
                    "message_id": "om_1",
                    "message_type": "text",
                    "content": json.dumps({"text": "你好，XingClaw"}),
                },
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_type": "user"},
            },
        }
        result = adapter.handle_webhook({}, json.dumps(payload).encode("utf-8"))
        self.assertEqual(result.ack.get("code"), 0)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].channel_id, "oc_x1")
        self.assertEqual(result.messages[0].text, "你好，XingClaw")
        self.assertIsNone(result.messages[0].thread_id)

    def test_parse_post_message_with_inline_code(self) -> None:
        adapter = FeishuAdapter(FeishuAdapterConfig(app_id="a", app_secret="b"))
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_id": "oc_x1", "message_id": "om_post", "message_type": "post",
                    "content": json.dumps({"zh_cn": {"title": "", "content": [[
                        {"tag": "text", "text": "test_color", "style": ["code"]},
                        {"tag": "text", "text": " 对应的记忆是什么？"},
                    ]]}}, ensure_ascii=False),
                },
                "sender": {"sender_id": {"open_id": "ou_1"}, "sender_type": "user"},
            },
        }
        result = adapter.handle_webhook({}, json.dumps(payload).encode("utf-8"))
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].text, "test_color 对应的记忆是什么？")

    def test_invalid_verify_token_is_rejected(self) -> None:
        adapter = FeishuAdapter(FeishuAdapterConfig(app_id="a", app_secret="b", verify_token="vt"))
        payload = {"token": "wrong", "header": {"event_type": "im.message.receive_v1"}, "event": {}}
        result = adapter.handle_webhook({}, json.dumps(payload).encode("utf-8"))
        self.assertEqual(result.ack.get("code"), 19021)
        self.assertEqual(len(result.messages), 0)

    def test_ignore_bot_sender_message(self) -> None:
        adapter = FeishuAdapter(FeishuAdapterConfig(app_id="a", app_secret="b", verify_token="vt"))
        payload = {
            "token": "vt",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_id": "oc_x1",
                    "message_id": "om_1",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_id": {"open_id": "ou_bot"}, "sender_type": "app"},
            },
        }
        result = adapter.handle_webhook({}, json.dumps(payload).encode("utf-8"))
        self.assertEqual(result.ack.get("code"), 0)
        self.assertEqual(len(result.messages), 0)


if __name__ == "__main__":
    unittest.main()
