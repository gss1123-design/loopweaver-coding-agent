from __future__ import annotations

"""
Anthropic Messages API 流式 provider。

实现思路：
1) 读取 SSE 的 event/data；
2) 按 content block 组装 text/thinking/toolCall；
3) 映射 stop reason 并输出统一 done/error 事件。
"""

import asyncio
import json
from typing import Any

import httpx

from ..env_api_keys import get_env_api_key
from ..event_stream import AssistantMessageEventStream
from ..cancellation import throw_if_cancelled
from ..types import Context, Model, SimpleStreamOptions, StreamOptions, TextContent, ThinkingContent, ToolCall
from ._common import empty_assistant_message, parse_partial_json, to_anthropic_messages, to_anthropic_tools


def _map_stop_reason(reason: str | None) -> str:
    if reason == "tool_use":
        return "toolUse"
    if reason == "max_tokens":
        return "length"
    return "stop"


def stream_anthropic(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    resolved_options = options or StreamOptions()
    stream = AssistantMessageEventStream(signal=resolved_options.signal)

    async def _run() -> None:
        out = empty_assistant_message(api=model.api, provider=model.provider, model=model.id)
        try:
            throw_if_cancelled(resolved_options.signal)
            api_key = resolved_options.api_key or get_env_api_key(model.provider)
            if not api_key:
                raise RuntimeError("Missing ANTHROPIC_API_KEY")

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            if model.headers:
                headers.update(model.headers)
            if resolved_options.headers:
                headers.update(resolved_options.headers)

            payload: dict[str, Any] = {
                "model": model.id,
                "max_tokens": resolved_options.max_tokens or model.max_tokens,
                "messages": to_anthropic_messages(context),
                "stream": True,
            }
            if context.system_prompt:
                payload["system"] = context.system_prompt
            if resolved_options.temperature is not None:
                payload["temperature"] = resolved_options.temperature
            tools = to_anthropic_tools(context.tools)
            if tools:
                payload["tools"] = tools

            timeout = resolved_options.timeout_seconds or None
            # 建立异步 HTTP 客户端,设置超时(默认 120 秒)
            async with httpx.AsyncClient(timeout=timeout) as client:
                throw_if_cancelled(resolved_options.signal)
                # 发起流式 POST 请求,注意用 client.stream 而不是 client.post
                # stream 方法返回的 response 支持按行迭代,不会一次性加载全部响应体
                async with client.stream(
                    "POST",
                    # 组装完整 URL:去掉 base_url 末尾的斜杠,拼 /v1/messages
                    # 例如 base_url="https://api.anthropic.com/" → "https://api.anthropic.com/v1/messages"
                    f"{model.base_url.rstrip('/')}/v1/messages",
                    headers=headers,  # 包含 x-api-key、anthropic-version、content-type 等
                    json=payload,     # 请求体:model、messages、tools、system 等,httpx 自动序列化成 JSON
                ) as response:
                    # 如果 HTTP 状态码 >= 400,抛 httpx.HTTPStatusError
                    # 例如 401(无效 key)、429(超限)、500(服务端错误)
                    response.raise_for_status()
                    throw_if_cancelled(resolved_options.signal)

                    # 推第一个事件:流开始
                    # out 此时只有空框架(response_id 未填、content=[]、usage=0/0)
                    # 主循环收到这个事件后会把 out 塞进 context.messages 做占位
                    stream.push({"type": "start", "partial": out})

                    # === SSE 流解析状态机 ===
                    # Anthropic 的 SSE 格式是:
                    #   event: message_start
                    #   data: {"type":"message_start","message":{...}}
                    #
                    #   event: content_block_start
                    #   data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}
                    #
                    #   event: content_block_delta
                    #   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你"}}
                    #   ... (多次 delta,每次追加一小段)
                    #
                    #   event: content_block_stop
                    #   data: {"type":"content_block_stop","index":0}
                    #
                    #   event: message_delta
                    #   data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{...}}
                    #
                    # 我们要追踪当前在处理哪个事件、哪个 content block

                    current_event: str | None = None   # 当前 SSE 事件名(message_start/content_block_delta/...)
                    current_index: int | None = None   # Anthropic 的 content 块索引(一条回复可能有多个块)

                    # === 内容块缓存:按 index 存放正在构建的块对象 ===
                    # 为什么要分三个字典?因为 Anthropic API 的响应是增量的:
                    # 先发 content_block_start 告诉你"有个文本块",然后多次发 content_block_delta 追加内容
                    # 我们要把同一个 index 的所有 delta 累加到同一个对象上
                    text_blocks: dict[int, TextContent] = {}       # index → 文本块
                    thinking_blocks: dict[int, ThinkingContent] = {}  # index → 思考块(<thinking>标签内容)
                    tool_blocks: dict[int, ToolCall] = {}          # index → 工具调用块
                    tool_partial_json: dict[int, str] = {}         # index → 工具参数的部分 JSON 字符串

                    # === 逐行读取 SSE 响应流 ===
                    # response.aiter_lines() 是异步生成器,每次 yield 一行
                    # 遇到网络没数据时就 await,让位给事件循环,等网络传来下一批数据
                    async for raw_line in response.aiter_lines():
                        throw_if_cancelled(resolved_options.signal)
                        line = raw_line.strip()  # 去掉首尾空白
                        if not line:             # 空行跳过(SSE 用空行分隔事件)
                            continue

                        # SSE 的两种行格式:
                        # event: xxx     ← 事件名
                        # data: {...}    ← JSON 数据
                        if line.startswith("event:"):
                            # 提取事件名,存到 current_event,继续读下一行
                            # 下一行的 data 就属于这个事件
                            current_event = line[len("event:") :].strip()
                            continue
                        if not line.startswith("data:"):
                            # 既不是 event 也不是 data,跳过(可能是注释行或其他)
                            continue

                        # 解析 data 行:去掉 "data:" 前缀,剩下的是 JSON
                        # 解析 data 行:去掉 "data:" 前缀,剩下的是 JSON
                        data = json.loads(line[len("data:") :].strip())

                        # === 根据事件类型分发处理 ===
                        # Anthropic 的事件序列:
                        # message_start → content_block_start → 多次 content_block_delta → content_block_stop → message_delta

                        if current_event == "message_start":
                            # 流的开头,拿到完整消息的元数据(id、初始 usage)
                            message = data.get("message", {})
                            out.response_id = message.get("id")  # 例如 "msg_01abc123"
                            usage = message.get("usage", {})
                            # input_tokens 在 message_start 就给了(因为请求已经发完,API 知道你发了多少 token)
                            # input_tokens 在 message_start 就给了(因为请求已经发完,API 知道你发了多少 token)
                            out.usage.input = usage.get("input_tokens", out.usage.input)

                        elif current_event == "content_block_start":
                            # 新的内容块开始(可能是文本、思考、工具调用)
                            # data 格式: {"index": 0, "content_block": {"type": "text"}}
                            current_index = data.get("index", 0)  # 这个块的索引,从 0 开始
                            block = data.get("content_block", {})
                            block_type = block.get("type")  # "text" | "thinking" | "redacted_thinking" | "tool_use"

                            if block_type == "text":
                                # 创建一个空的文本块对象
                                tb = TextContent(text="")
                                text_blocks[current_index] = tb  # 存到字典,后续 delta 会往这里追加
                                out.content.append(tb)  # 同时加进 out.content,保持引用一致
                                # 推事件通知主循环:"第 X 个块开始了,是文本"
                                stream.push(
                                    {"type": "text_start", "contentIndex": len(out.content) - 1, "partial": out}
                                )

                            elif block_type in {"thinking", "redacted_thinking"}:
                                # 思考块(<thinking>标签内容),redacted 表示被审查过的
                                th = ThinkingContent(thinking="", redacted=(block_type == "redacted_thinking"))
                                thinking_blocks[current_index] = th
                                out.content.append(th)
                                stream.push(
                                    {"type": "thinking_start", "contentIndex": len(out.content) - 1, "partial": out}
                                )

                            elif block_type == "tool_use":
                                # 工具调用块,此时已经知道 id 和 name,但参数还没来
                                tc = ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments={})
                                tool_blocks[current_index] = tc
                                tool_partial_json[current_index] = ""  # 初始化 JSON 累加器
                                out.content.append(tc)
                                stream.push(
                                    {"type": "toolcall_start", "contentIndex": len(out.content) - 1, "partial": out}
                                )

                        elif current_event == "content_block_delta":
                            # 内容块的增量更新(最频繁的事件)
                            # data 格式: {"index": 0, "delta": {"type": "text_delta", "text": "你"}}
                            idx = data.get("index", current_index if current_index is not None else 0)
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")  # "text_delta" | "thinking_delta" | "signature_delta" | "input_json_delta"

                            if delta_type == "text_delta" and idx in text_blocks:
                                # 文本增量:把新来的字符追加到对应的文本块上
                                text = delta.get("text", "")
                                text_blocks[idx].text += text  # 累加到同一个对象
                                # 推事件:"又来了一段文字"
                                # 注意 contentIndex 是从 out.content 里找这个块的位置
                                # 因为 out.content 可能混着文本、思考、工具调用多种块
                                stream.push(
                                    {
                                        "type": "text_delta",
                                        "contentIndex": out.content.index(text_blocks[idx]),
                                        "delta": text,
                                        "partial": out,
                                    }
                                )

                            elif delta_type in {"thinking_delta", "signature_delta"} and idx in thinking_blocks:
                                # 思考块增量(signature_delta 是签名,实际项目里没用到)
                                text = delta.get("thinking", "")
                                if text:
                                    thinking_blocks[idx].thinking += text
                                    stream.push(
                                        {
                                            "type": "thinking_delta",
                                            "contentIndex": out.content.index(thinking_blocks[idx]),
                                            "delta": text,
                                            "partial": out,
                                        }
                                    )

                            elif delta_type == "input_json_delta" and idx in tool_blocks:
                                # 工具参数的 JSON 增量
                                # Anthropic 一个字符一个字符发,我们累加成完整 JSON 字符串
                                piece = delta.get("partial_json", "")
                                tool_partial_json[idx] += piece
                                # parse_partial_json 尝试解析半截 JSON,解析不了就返回 {}
                                # 这个函数实际上只做完整解析,半截的会失败返回空字典
                                tool_blocks[idx].arguments = parse_partial_json(tool_partial_json[idx])
                                stream.push(
                                    {
                                        "type": "toolcall_delta",
                                        "contentIndex": out.content.index(tool_blocks[idx]),
                                        "delta": piece,
                                        "partial": out,
                                    }
                                )

                        elif current_event == "content_block_stop":
                            # 某个内容块结束了
                            # data 格式: {"index": 0}
                            idx = data.get("index", current_index if current_index is not None else 0)

                            if idx in text_blocks:
                                # 文本块结束,推 text_end 事件
                                block = text_blocks[idx]
                                stream.push(
                                    {
                                        "type": "text_end",
                                        "contentIndex": out.content.index(block),
                                        "content": block.text,  # 完整的文本
                                        "partial": out,
                                    }
                                )

                            elif idx in thinking_blocks:
                                # 思考块结束
                                block = thinking_blocks[idx]
                                stream.push(
                                    {
                                        "type": "thinking_end",
                                        "contentIndex": out.content.index(block),
                                        "content": block.thinking,
                                        "partial": out,
                                    }
                                )

                            elif idx in tool_blocks:
                                # 工具调用块结束
                                block = tool_blocks[idx]
                                stream.push(
                                    {
                                        "type": "toolcall_end",
                                        "contentIndex": out.content.index(block),
                                        "toolCall": block,  # 完整的工具调用对象
                                        "partial": out,
                                    }
                                )

                        elif current_event == "message_delta":
                            # 整条消息的元数据更新(通常在最后)
                            # data 格式: {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}}
                            delta = data.get("delta", {})
                            usage = data.get("usage", {})
                            # stop_reason 决定主循环要不要继续:
                            # "end_turn" → "stop"(说完了)
                            # "tool_use" → "toolUse"(要调工具)
                            # "max_tokens" → "length"(太长被截)
                            out.stop_reason = _map_stop_reason(delta.get("stop_reason"))
                            # output_tokens 到这里才知道(因为要生成完才能统计)
                            out.usage.output = usage.get("output_tokens", out.usage.output)

                    # === SSE 流结束,推最终事件 ===
                    # async for 循环退出说明所有事件都处理完了
                    stream.push({"type": "done", "reason": out.stop_reason, "message": out})
                    # 调用 end() 把完整消息装进 Future,让 await result() 能拿到
                    stream.end(out)

        except asyncio.CancelledError:
            stream.cancel()
            raise
        except Exception as exc:
            # 任何异常(网络错误、JSON 解析失败、HTTP 错误等)都在这里捕获
            out.stop_reason = "error"
            out.error_message = str(exc)  # 错误信息存进消息,主循环能看到
            stream.push({"type": "error", "reason": "error", "error": out})
            stream.end(out)  # 即使出错也要 end(),让 await result() 能返回

    # === 关键:后台启动 ===
    # 不用 await,立刻启动 _run() 作为后台任务
    # 这样 stream_anthropic() 能立刻返回 stream 对象,不阻塞主循环
    # _run() 在后台慢慢发 HTTP、解析响应、push 事件
    worker = asyncio.create_task(_run())
    stream.attach_task(worker)
    return stream


def stream_simple_anthropic(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    # 第一阶段实现：simple 接口复用标准 stream。
    return stream_anthropic(model, context, options)
