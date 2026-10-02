from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from im.feishu_longconn import _parse_card_action_object, _parse_event_object, _parse_ws_message


class FeishuLongConnTests(unittest.TestCase):
    def test_parse_card_approval_action(self) -> None:
        data = SimpleNamespace(
            event=SimpleNamespace(
                action=SimpleNamespace(
                    value={
                        "loopweaver_action": "tool_approval",
                        "decision": "reject",
                        "tool_call_id": "tc-2",
                        "thread_id": "omt_1",
                    }
                ),
                context=SimpleNamespace(open_chat_id="oc_1", open_message_id="om_card"),
                operator=SimpleNamespace(open_id="ou_owner", user_id=None),
            )
        )

        msg = _parse_card_action_object(data)

        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertEqual(msg.text, "/reject tc-2")
        self.assertEqual(msg.user_id, "ou_owner")
        self.assertEqual(msg.channel_id, "oc_1")
        self.assertEqual(msg.thread_id, "omt_1")

    def test_parse_text_event(self) -> None:
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_id": "oc_1",
                    "message_id": "om_1",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_1"}},
            },
        }
        msg = _parse_ws_message(payload)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertEqual(msg.channel_id, "oc_1")
        self.assertEqual(msg.message_id, "om_1")
        self.assertEqual(msg.thread_id, None)

    def test_parse_post_with_inline_code_and_group_type(self) -> None:
        body = json.dumps({"title": "", "content": [[
            {"tag": "text", "text": "test_color "},
            {"tag": "text", "text": "对应的记忆是什么？", "style": ["code"]},
        ]]}, ensure_ascii=False)
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {"chat_id": "oc_1", "message_id": "om_post", "message_type": "post",
                            "chat_type": "p2p", "content": body},
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_1"}},
            },
        }
        msg = _parse_ws_message(payload)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertEqual(msg.text, "test_color 对应的记忆是什么？")

        typed = SimpleNamespace(event=SimpleNamespace(
            message=SimpleNamespace(chat_id="oc_1", message_id="om_post", message_type="post",
                                    chat_type="group", content=body, root_id=None, create_time=None,
                                    mentions=[]),
            sender=SimpleNamespace(sender_type="user", sender_id=SimpleNamespace(open_id="ou_1")),
        ))
        typed_msg = _parse_event_object(typed)
        self.assertIsNotNone(typed_msg)
        assert typed_msg is not None
        self.assertEqual(typed_msg.text, "test_color 对应的记忆是什么？")
        self.assertEqual(typed_msg.raw["chat_type"], "group")

    def test_ignore_bot_message(self) -> None:
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_id": "oc_1",
                    "message_id": "om_1",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_type": "app", "sender_id": {"open_id": "ou_bot"}},
            },
        }
        self.assertIsNone(_parse_ws_message(payload))


if __name__ == "__main__":
    unittest.main()
