from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.providers._common import to_anthropic_messages, to_openai_messages
from ai.types import AssistantMessage, Context, TextContent, ToolCall, ToolResultMessage, UserMessage


class ProviderMessageSerializationTests(unittest.TestCase):
    def test_orphan_tool_chain_is_omitted(self) -> None:
        context = Context(
            messages=[
                UserMessage(content="run it"),
                AssistantMessage(
                    content=[ToolCall(id="missing-result", name="bash", arguments={})],
                    stop_reason="toolUse",
                ),
                ToolResultMessage(
                    tool_call_id="unknown-call",
                    content=[TextContent(text="orphan")],
                ),
                AssistantMessage(content=[TextContent(text="done")], stop_reason="stop"),
            ]
        )

        openai = to_openai_messages(context)
        anthropic = to_anthropic_messages(context)

        self.assertEqual([item["role"] for item in openai], ["user", "assistant"])
        self.assertEqual([item["role"] for item in anthropic], ["user", "assistant"])

    def test_complete_tool_chain_is_preserved(self) -> None:
        context = Context(
            messages=[
                UserMessage(content="run it"),
                AssistantMessage(
                    content=[ToolCall(id="tc1", name="bash", arguments={"command": "echo ok"})],
                    stop_reason="toolUse",
                ),
                ToolResultMessage(
                    tool_call_id="tc1",
                    content=[TextContent(text="ok")],
                ),
            ]
        )

        openai = to_openai_messages(context)
        anthropic = to_anthropic_messages(context)

        self.assertEqual([item["role"] for item in openai], ["user", "assistant", "tool"])
        self.assertEqual([item["role"] for item in anthropic], ["user", "assistant", "user"])
        self.assertEqual(openai[-1]["tool_call_id"], "tc1")


if __name__ == "__main__":
    unittest.main()
