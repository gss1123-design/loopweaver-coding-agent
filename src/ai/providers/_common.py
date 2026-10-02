from __future__ import annotations

"""
provider 共享工具函数：
1) 通用消息转换（Context -> provider payload）
2) 流式 JSON 片段解析
3) 空 AssistantMessage 初始化
"""

import json
import time
from typing import Any

from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_partial_json(raw: str) -> dict[str, Any]:
    """
    解析流式工具参数（可能是半截 JSON）。
    解析失败时返回 {}，让上层保持稳态。
    """
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def empty_assistant_message(api: str, provider: str, model: str) -> AssistantMessage:
    """创建一个最小可用的 AssistantMessage，用于边流式边填充。"""
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model,
        usage=Usage(),
        timestamp=now_ms(),
    )


def to_openai_messages(context: Context) -> list[dict[str, Any]]:
    """把统一 Message 转成 OpenAI Chat Completions 的 messages。"""
    out: list[dict[str, Any]] = []
    known_tool_call_ids: set[str] = set()
    if context.system_prompt:
        out.append({"role": "system", "content": context.system_prompt})
    for index, msg in enumerate(context.messages):
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{part.mime_type};base64,{part.data}"},
                            }
                        )
                out.append({"role": "user", "content": parts})
        elif isinstance(msg, AssistantMessage):
            # 失败请求产生的空 assistant 只用于记录错误，不应再次发给 provider。
            # 某些 OpenAI 兼容网关会把空 content 判为 400 Bad Request。
            if not msg.content:
                continue
            text = "".join(b.text for b in msg.content if isinstance(b, TextContent))
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            # OpenAI requires one tool result for every assistant tool call.
            # Interrupted/crashed sessions can leave a half-written chain;
            # omit that assistant message instead of sending a request that
            # the provider will reject with HTTP 400.
            result_ids_after = {
                item.tool_call_id
                for item in context.messages[index + 1 :]
                if isinstance(item, ToolResultMessage) and item.tool_call_id
            }
            if tool_calls and not all(
                call.id and call.id in result_ids_after for call in tool_calls
            ):
                continue
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": text if text else (None if tool_calls else ""),
            }
            if tool_calls:
                known_tool_call_ids.update(tc.id for tc in tool_calls if tc.id)
                payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in tool_calls
                ]
            out.append(payload)
        elif isinstance(msg, ToolResultMessage):
            # 历史文件可能残留没有对应 assistant ToolCall 的结果；跳过它，
            # 否则 OpenAI 兼容接口通常会返回 400。
            if not msg.tool_call_id or msg.tool_call_id not in known_tool_call_ids:
                continue
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": "\n".join(
                        p.text for p in msg.content if isinstance(p, TextContent) and isinstance(p.text, str)
                    ),
                }
            )
    return out


def to_openai_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """把统一 Tool 定义转成 OpenAI tools。"""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def to_anthropic_messages(context: Context) -> list[dict[str, Any]]:
    """把统一 Message 转成 Anthropic Messages API payload。"""
    out: list[dict[str, Any]] = []
    known_tool_call_ids: set[str] = set()
    for index, msg in enumerate(context.messages):
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": part.mime_type, "data": part.data},
                            }
                        )
                out.append({"role": "user", "content": parts})
        elif isinstance(msg, AssistantMessage):
           # 先找出这条消息里所有的 ToolCall
            tool_calls = [block for block in msg.content if isinstance(block, ToolCall)]

            # 往后扫:这些 ToolCall 在后续消息里有没有对应的 ToolResult?
            # context.messages[index + 1:] 是当前消息之后的所有消息
            result_ids_after = {
                item.tool_call_id
                for item in context.messages[index + 1:]
                if isinstance(item, ToolResultMessage) and item.tool_call_id
            }

            # 如果这条消息有 ToolCall,但不是所有 ToolCall 都有对应的 Result:
            # 说明这是一条"半截"消息(上下文压缩 bug 会产生这种情况),跳过它
            # 发给 API 一条没有 Result 的 ToolCall 会被拒绝,所以直接丢掉更安全
            if tool_calls and not all(
                call.id and call.id in result_ids_after for call in tool_calls
            ):
                continue  # 跳过这条,不加进 out
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolCall):
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        }
                    )
            known_tool_call_ids.update(call.id for call in tool_calls if call.id)
            out.append({"role": "assistant", "content": parts})
        elif isinstance(msg, ToolResultMessage):
            if not msg.tool_call_id or msg.tool_call_id not in known_tool_call_ids:
                continue
            text = "\n".join(p.text for p in msg.content if isinstance(p, TextContent))
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": [{"type": "text", "text": text}],
                            "is_error": msg.is_error,
                        }
                    ],
                }
            )
    return out


def to_anthropic_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    """把统一 Tool 定义转成 Anthropic tools。"""
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]
