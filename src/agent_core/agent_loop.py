from __future__ import annotations

"""
Agent 主循环实现：
用户消息 -> LLM -> 工具调用 -> LLM -> ... -> 结束

这是整个 agent 的心脏,负责协调 LLM 和工具之间的来回对话。
"""

import asyncio
import posixpath
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, cast

from ai.stream import stream_simple
from ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

from .cancellation import await_with_cancellation, throw_if_cancelled
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
)


# 流式函数的类型签名,用于支持测试时注入假 provider
StreamFn = Callable[[Any, Context, SimpleStreamOptions | None], Any | Awaitable[Any]]


def _now_ms() -> int:
    """返回当前毫秒时间戳,用于给事件和消息打时间标记"""
    return int(time.time() * 1000)


def _error_tool_result(message: str) -> AgentToolResult:
    """构造一个错误的工具结果,当工具找不到或执行失败时使用"""
    return AgentToolResult(content=[TextContent(text=message)], details={})


async def _maybe_await(value: Any) -> Any:
    """
    如果传入的是协程或 Future 就 await,否则原样返回。

    这个函数在项目里到处用,因为钩子函数可能是普通函数也可能是 async 函数,
    调用方不想关心到底是哪种,统一用这个包一下。
    """
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


async def _emit(emit: AgentEventSink, event: dict[str, Any]) -> None:
    """推送一个事件,支持 emit 是同步或异步函数"""
    await _maybe_await(emit(cast(AgentEvent, event)))


def _with_event_schema(
    emit: AgentEventSink,
    session_id: str | None,
    operation_id: str | None = None,
    attempt: int = 1,
) -> AgentEventSink:
    """
    给事件流包装元数据的装饰器。

    每个事件自动加上:
    - runId: 本次 run 唯一 ID (格式 run_abc123...)
    - turnId: 当前轮次(从 0 开始,每遇到 turn_start 事件就 +1)
    - eventId: 本次 run 内递增的事件 ID (格式 run_abc:1, run_abc:2...)
    - timestamp: 事件毫秒时间戳
    - sessionId: 透传上层 session ID

    为什么需要这些?因为事件流可能跨越多轮、多个工具调用,
    有了这些元数据就能在日志里追踪"第几轮发生了什么、用了多少时间"。
    """

    # 生成唯一的 run ID,12 位十六进制
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    turn_id = 0      # 轮次计数,第一轮是 1(遇到 turn_start 后才 +1)
    event_seq = 0    # 事件序号,从 1 开始

    async def _wrapped(event: dict[str, Any]) -> None:
        # nonlocal 声明:要修改外层函数的变量,必须显式声明
        # 否则 turn_id += 1 会被当成创建新的局部变量
        nonlocal turn_id, event_seq
        event_type = event.get("type")

        # 遇到 turn_start 就增加轮次
        if event_type == "turn_start":
            turn_id += 1

        # 每个事件都递增序号
        event_seq += 1

        # 用字典解包 {**event, ...} 保留原事件所有字段,再加新字段
        enriched = {
            **event,
            "runId": run_id,
            "turnId": turn_id,
            "eventId": f"{run_id}:{event_seq}",
            "timestamp": _now_ms(),
            "sessionId": session_id,
        }
        if operation_id:
            enriched["operationId"] = operation_id
        if attempt > 0:
            enriched["attempt"] = int(attempt)
        await _maybe_await(emit(cast(AgentEvent, enriched)))

    return _wrapped


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """
    主循环入口:处理用户的新消息。

    参数:
    - prompts: 用户新说的话(一条或多条)
    - context: 包含历史消息、system prompt、可用工具
    - config: 配置(模型、钩子、执行策略等)
    - emit: 事件监听器,所有事件都推给它
    - signal: 可选的协作式取消信号，会传给工具、钩子和 provider
    - stream_fn: 测试用,可以注入假的 provider

    返回: 本次 run 新增的所有消息(包括用户消息、assistant 回复、工具结果)
    """
    throw_if_cancelled(signal)

    # 包装 emit,给所有事件加上 runId/turnId/eventId/timestamp
    emit = _with_event_schema(emit, config.session_id, config.operation_id, config.attempt)

    # new_messages 记录这次 run 产生的所有新消息
    new_messages: list[AgentMessage] = list(prompts)

    # current_context 是"当前上下文",包含历史 + 新消息
    # 模型看到的就是这个 context.messages
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],  # 历史 + 新消息
        tools=context.tools,
    )

    # 推送开始事件
    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})

    # 为每条用户消息推送 message_start/message_end
    # 让监听器知道"用户说了什么"
    for prompt in prompts:
        await _emit(emit, {"type": "message_start", "message": prompt})
        await _emit(emit, {"type": "message_end", "message": prompt})

    # 进入真正的循环:调模型 -> 执行工具 -> 再调模型 -> ...
    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)

    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """
    继续已有对话:不添加新的用户消息,直接让模型处理现有上下文。

    典型场景:上下文最后一条是尚未处理的 UserMessage，或工具执行完成后
    已追加 ToolResultMessage，需要让模型进入下一轮。若最后一条是因
    max_tokens 截断的 AssistantMessage，聊天 API 通常不允许直接续接；
    应由上层添加一条明确的续写用户消息后再调用普通 prompt 流程。

    参数:
    - context: 必须包含历史消息,且最后一条不能是 assistant 消息
    - 其他参数同 run_agent_loop

    返回: 本次续跑产生的新消息
    """
    # 检查前置条件
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")
    throw_if_cancelled(signal)

    emit = _with_event_schema(emit, config.session_id, config.operation_id, config.attempt)
    new_messages: list[AgentMessage] = []  # 注意这里是空的,不添加用户消息
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),  # 复制一份历史消息
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})

    await _run_loop(current_context, new_messages, config, emit, signal, stream_fn)
    return new_messages


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> None:
    """
    真正的主循环:调模型 -> 执行工具 -> 再调模型 -> ... 直到结束。

    循环结束的三种情况:
    1. 模型回复不再包含工具调用(说完了)
    2. 模型返回 error 或 aborted
    3. 达到最大轮数限制

    关键变量:
    - first_turn: 是否第一轮(第一轮不推 turn_start,因为外面已经推过了)
    - turns: 轮数计数,用于判断是否超限
    - pending_messages: 待插入的"操舵消息"(用户在运行中追加的指令)
    - has_more_tool_calls: 模型是否还要调工具
    """
    first_turn = True
    turns = 0

    throw_if_cancelled(signal)

    # 检查是否有"操舵消息"(用户运行中插入的新指令)
    # get_steering_messages 是可选的回调,CLI/IM 层通常不提供
    pending_messages = await _maybe_await(config.get_steering_messages()) if config.get_steering_messages else []

    # 最大轮数,至少是 1
    max_turns = max(1, int(config.max_turns))

    while True:
        throw_if_cancelled(signal)
        has_more_tool_calls = True

        # 内层循环:只要有工具调用或有待插入的消息,就继续跑
        while has_more_tool_calls or pending_messages:
            throw_if_cancelled(signal)
            # 检查是否超过最大轮数
            if turns >= max_turns:
                await _emit(emit, {"type": "max_turns_reached", "turns": turns})
                await _emit(emit, {"type": "agent_end", "messages": new_messages})
                return

            turns += 1

            # 第一轮不推 turn_start(外面已经推过了),后续轮次才推
            if not first_turn:
                await _emit(emit, {"type": "turn_start"})
            else:
                first_turn = False

            # 如果有待插入的消息,先插进去
            # 典型场景:用户在模型跑的过程中补充了新指令
            if pending_messages:
                for message in pending_messages:
                    throw_if_cancelled(signal)
                    await _emit(emit, {"type": "message_start", "message": message})
                    await _emit(emit, {"type": "message_end", "message": message})
                    current_context.messages.append(message)  # 加进上下文,模型下轮能看到
                    new_messages.append(message)
                pending_messages = []  # 清空,等下次再问

            # === 核心步骤 1: 调用模型,获取回复 ===
            assistant = await _stream_assistant_response(current_context, config, emit, signal, stream_fn)
            throw_if_cancelled(signal)
            new_messages.append(assistant)

            # === 核心步骤 2: 检查是否出错或中断 ===
            if assistant.stop_reason in {"error", "aborted"}:
                # 出错了,立刻结束整个循环
                await _emit(emit, {"type": "turn_end", "message": assistant, "toolResults": []})
                await _emit(emit, {"type": "agent_end", "messages": new_messages})
                return

            # === 核心步骤 3: 从回复里挑出工具调用 ===
            # assistant.content 是个列表,里面混着 TextContent 和 ToolCall
            # 只要 ToolCall
            tool_calls = [c for c in assistant.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0
            tool_results: list[ToolResultMessage] = []

            # === 核心步骤 4: 如果有工具调用,执行它们 ===
            if has_more_tool_calls:
                throw_if_cancelled(signal)
                tool_results = await _execute_tool_calls(current_context, assistant, config, emit, signal)
                # 把工具结果加进上下文和 new_messages
                # 这样模型下一轮能看到"你调了什么、结果是什么"
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            # 推送本轮结束事件
            await _emit(emit, {"type": "turn_end", "message": assistant, "toolResults": tool_results})

            # 再次检查是否有新的操舵消息(用户可能又补充了指令)
            pending_messages = await _maybe_await(config.get_steering_messages()) if config.get_steering_messages else []
            throw_if_cancelled(signal)

        # === 外层循环:检查是否有后续任务 ===
        # get_follow_up_messages 是可选回调,通常也不提供
        # 如果有,就把它们当成 pending_messages 继续跑
        followups = await _maybe_await(config.get_follow_up_messages()) if config.get_follow_up_messages else []
        throw_if_cancelled(signal)
        if followups:
            pending_messages = followups
            continue  # 继续外层循环
        break  # 没有后续任务,真正退出

    await _emit(emit, {"type": "agent_end", "messages": new_messages})


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    """
    调用模型并流式接收回复。

    流程:
    1. 可选的上下文转换(transform_context):让钩子函数能修改消息列表
    2. 转换成 LLM 层的消息格式(convert_to_llm)
    3. 调用 stream_simple(或测试注入的 stream_fn)获取事件流
    4. 监听流里的每个事件,转发给 emit
    5. 最后拿到完整的 AssistantMessage 返回

    关键点:
    - partial 是"正在构建中的消息",每次 delta 事件都更新它
    - added_partial 标记是否已经把 partial 加进 context.messages
      (只在 start 事件加一次,后续 delta 直接更新同一个对象)
    """
    messages = context.messages
    throw_if_cancelled(signal)

    # 可选的上下文转换:钩子函数可以修改消息列表
    # 例如:合并连续的用户消息、截断太长的历史等
    if config.transform_context:
        messages = await _maybe_await(config.transform_context(messages, signal))
        throw_if_cancelled(signal)

    # 转换成 LLM 层认的消息格式
    # AgentMessage 可能有些自定义字段,LLM 层只认标准的 Message
    llm_messages = await _maybe_await(config.convert_to_llm(messages))
    throw_if_cancelled(signal)

    # 构造 LLM 上下文
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,  # AgentTool 与 ai.Tool 字段兼容,可以直接传
    )

    # 动态获取 API key(支持多 provider 场景)
    resolved_api_key = config.get_api_key and await _maybe_await(config.get_api_key(config.model.provider))
    throw_if_cancelled(signal)

    # 流式选项
    options = SimpleStreamOptions(
        reasoning=config.reasoning,          # Extended Thinking 模式开关
        api_key=resolved_api_key,
        session_id=config.session_id,
        max_tokens=config.max_tokens,
        signal=signal,
    )

    # 调用流式函数:测试时用 stream_fn,正常运行用 stream_simple
    fn = stream_fn or stream_simple
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    request_started = time.perf_counter()
    request_started_ms = _now_ms()
    chunk_count = 0
    first_chunk_at: float | None = None
    request_ended = False
    stream_error_event = False

    await _emit(
        emit,
        {
            "type": "ai_request_start",
            "requestId": request_id,
            "provider": str(config.model.provider),
            "model": str(config.model.id),
            "api": str(config.model.api),
            "streaming": True,
            "timestamp": request_started_ms,
        },
    )

    async def finish_request(
        message: AssistantMessage | None = None,
        *,
        error_type: str | None = None,
    ) -> None:
        """Emit exactly one provider-request terminal event."""

        nonlocal request_ended
        if request_ended:
            return
        request_ended = True
        stop_reason = message.stop_reason if message is not None else None
        is_error = bool(stream_error_event or stop_reason in {"error", "aborted"} or error_type)
        await _emit(
            emit,
            {
                "type": "ai_request_end",
                "requestId": request_id,
                "provider": str(config.model.provider),
                "model": str(config.model.id),
                "api": str(config.model.api),
                "streaming": True,
                "durationMs": max(0, int((time.perf_counter() - request_started) * 1000)),
                "chunkCount": chunk_count,
                "timeToFirstChunkMs": (
                    max(0, int((first_chunk_at - request_started) * 1000))
                    if first_chunk_at is not None
                    else None
                ),
                "isError": is_error,
                "errorType": error_type,
                "responseId": message.response_id if message is not None else None,
                "stopReason": stop_reason,
                # This is an in-memory event only; RunTraceRecorder extracts
                # usage and never persists the complete response here.
                "message": message,
            },
        )

    try:
        response_stream = await _maybe_await(fn(config.model, llm_context, options))
        throw_if_cancelled(signal)

        # partial 是正在构建中的消息,一开始是 None
        partial: AssistantMessage | None = None
        # added_partial 标记是否已经把 partial 塞进 context.messages
        # 只在第一次(start 事件)塞,后续 delta 直接更新同一个对象
        added_partial = False

        async def finish_response(final_message: AssistantMessage) -> AssistantMessage:
            if added_partial:
                # 如果已经加过占位消息,就替换它
                context.messages[-1] = final_message
            else:
                # 如果没加过(理论上不会发生,但防御性编程),就追加
                context.messages.append(final_message)
                await _emit(emit, {"type": "message_start", "message": final_message})

            await finish_request(final_message)
            await _emit(emit, {"type": "message_end", "message": final_message})
            return final_message

        # 监听流里的每个事件
        async for event in response_stream:
            throw_if_cancelled(signal)
            chunk_count += 1
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            t = event.get("type")

            if t == "start":
                # 流开始:拿到占位消息(response_id 有了,content 还是空的)
                partial = event["partial"]
                # 立刻把占位消息加进 context.messages
                # 这样后续 delta 事件直接更新同一个对象,不用每次都替换
                context.messages.append(partial)
                added_partial = True
                # 推送消息开始事件
                await _emit(emit, {"type": "message_start", "message": partial})

            elif t in {
                "text_start",       # 文本块开始
                "text_delta",       # 文本增量(模型又输出了几个字)
                "text_end",         # 文本块结束
                "thinking_start",   # 思考块开始(Extended Thinking)
                "thinking_delta",   # 思考增量
                "thinking_end",     # 思考块结束
                "toolcall_start",   # 工具调用块开始
                "toolcall_delta",   # 工具调用增量(JSON 参数一点点推过来)
                "toolcall_end",     # 工具调用块结束
            }:
                # 这些都是增量事件:event["partial"] 是更新后的完整消息
                # 我们把它赋给 partial,这样 context.messages 里的那个对象也跟着更新
                # (因为 Python 的引用语义,context.messages[-1] 就是 partial)
                if partial is not None:
                    partial = event["partial"]
                    # 更新 context.messages 里的那个占位消息
                    context.messages[-1] = partial
                    # 推送更新事件,让外层监听器能看到实时输出
                    await _emit(emit, {"type": "message_update", "message": partial, "assistantMessageEvent": event})

            elif t in {"done", "error"}:
                if t == "error":
                    stream_error_event = True
                # 流结束或出错:拿到最终消息
                final_message = await response_stream.result()
                throw_if_cancelled(signal)
                return await finish_response(final_message)

        # 如果 async for 正常退出(没遇到 done/error 就退出,理论上不会发生)
        final_message = await response_stream.result()
        throw_if_cancelled(signal)
        return await finish_response(final_message)
    except BaseException as exc:
        await finish_request(error_type=type(exc).__name__)
        raise


@dataclass
class _PreparedToolCall:
    """准备好的工具调用:找到了工具、解析了参数、通过了审批"""
    tool_call: ToolCall
    tool: AgentTool
    args: dict[str, Any]


@dataclass
class _ExecutedToolCall:
    """执行完的工具调用:结果 + 是否出错"""
    result: AgentToolResult
    is_error: bool


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    """
    执行所有工具调用,返回工具结果消息列表。

    根据 config.tool_execution 和工具安全属性决定串行还是并行:
    - "sequential": 始终一个一个执行；
    - "parallel": 只有全部工具明确只读、目标路径互不重叠时才并发，
      否则自动降级为串行。
    """
    # 从 assistant_message.content 里挑出所有 ToolCall
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]

    # 显式串行永远串行；parallel 是“允许安全并行”，不是强制所有工具并行。
    if config.tool_execution == "sequential" or _requires_sequential_execution(current_context, tool_calls):
        return await _execute_tool_calls_sequential(current_context, assistant_message, tool_calls, config, emit, signal)
    return await _execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, emit, signal)


def _requires_sequential_execution(
    current_context: AgentContext,
    tool_calls: list[ToolCall],
) -> bool:
    """判断 parallel 模式下的一批工具是否必须保守地串行执行。

    自动并行必须同时满足：
    1. 每个 ToolCall 都能找到对应 AgentTool；
    2. 每个工具都显式声明 ``read_only=True``；
    3. 每个调用的 ``arguments.path`` 目标互不相同、也不存在父子目录重叠。

    缺少 path 时按工作区根目录 ``.`` 处理。这会让无法证明资源互不冲突的
    工具保守串行，而不是冒险并发。
    """
    if len(tool_calls) <= 1:
        return False

    tools_by_name = {tool.name: tool for tool in current_context.tools}
    resource_paths: list[str] = []

    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call.name)
        if tool is None or not tool.read_only:
            return True

        raw_path = tool_call.arguments.get("path", ".")
        if not isinstance(raw_path, str):
            return True
        resource_paths.append(_normalize_resource_path(raw_path))

    for index, path in enumerate(resource_paths):
        for other in resource_paths[index + 1 :]:
            if _resource_paths_overlap(path, other):
                return True
    return False


def _normalize_resource_path(path: str) -> str:
    """把工具参数中的相对路径规范成仅用于冲突比较的统一形式。"""
    normalized = posixpath.normpath(path.strip().replace("\\", "/") or ".")
    # 所有路径挂到同一个虚拟根下，使 ``.`` 与 ``src/a.py`` 能识别为父子关系。
    return posixpath.normpath("/" + normalized.lstrip("/")).casefold()


def _resource_paths_overlap(left: str, right: str) -> bool:
    """相同路径或父子路径视为重叠，例如 ``src`` 与 ``src/app.py``。"""
    common = posixpath.commonpath([left, right])
    return common == left or common == right


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> tuple[_PreparedToolCall | None, AgentToolResult, bool]:
    """
    准备一个工具调用:查找工具、解析参数、调用 before_tool_call 钩子、等待审批。

    返回:
    - 成功: (_PreparedToolCall, 无意义的结果, False)
    - 失败: (None, 错误结果, True)

    失败的情况:
    1. 工具找不到
    2. before_tool_call 钩子阻止了执行
    3. 用户拒绝了审批
    """
    throw_if_cancelled(signal)

    # 第一步:查找工具
    tool = next((t for t in current_context.tools if t.name == tool_call.name), None)
    if tool is None:
        # 工具不存在,返回错误
        return None, _error_tool_result(f"Tool {tool_call.name} not found"), True

    # 第二步:解析参数(防御性:确保是字典)
    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}

    # 第三步:调用 before_tool_call 钩子(可选)
    # 钩子可以做权限检查、日志记录、参数验证等
    if config.before_tool_call:
        before = await _maybe_await(
            config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=args,
                    context=current_context,
                ),
                signal,
            )
        )
        throw_if_cancelled(signal)
        # 如果钩子返回 block=True,就阻止执行
        if before and before.block:
            return None, _error_tool_result(before.reason or "Tool execution was blocked"), True

    # 第四步:等待审批(可选)
    # approval_gate 是一个全局可插拔的"审批门"；是否真的需要等待，
    # 由当前 AgentTool.requires_approval 决定。这样只读工具可以在同一个
    # session 开启审批功能时自动放行，而写文件、执行命令等工具仍需确认。
    if config.approval_gate is not None and tool.requires_approval:
        id_for = getattr(config.approval_gate, "id_for", None)
        approval_id = id_for(tool_call.id) if callable(id_for) else tool_call.id
        request = {
            "tool_call_id": approval_id,
            "tool_name": tool_call.name,
            "args": args,
        }
        # 开始审批流程
        config.approval_gate.begin(request)
        # 推送"需要审批"事件,让 UI 能弹出确认对话框
        await _emit(
            emit,
            {
                "type": "approval_required",
                "toolCallId": approval_id,
                "toolName": tool_call.name,
                "args": args,
            },
        )
        # 等待用户审批(approval_gate.wait 会阻塞直到用户点击"允许"或"拒绝")
        approved = await await_with_cancellation(
            _maybe_await(config.approval_gate.wait(approval_id)),
            signal,
        )
        if not approved:
            # 用户拒绝了或超时,返回错误
            return None, _error_tool_result("Tool execution was rejected or timed out"), True

    # 所有检查都通过,返回准备好的工具调用
    return _PreparedToolCall(tool_call=tool_call, tool=tool, args=args), AgentToolResult(content=[]), False


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    emit: AgentEventSink,
    signal: Any | None,
) -> _ExecutedToolCall:
    """
    执行一个已准备好的工具调用。

    流程:
    1. 调用 tool.execute(传入 on_update 回调)
    2. 工具执行过程中可能多次调用 on_update 推送进度
    3. 捕获异常,转成错误结果

    返回: _ExecutedToolCall(结果, 是否出错)
    """
    throw_if_cancelled(signal)
    # Persist execution intent before entering the tool. A checkpoint failure
    # must propagate, not be converted into a successful-to-run tool error.
    await _emit(emit, {
        "type": "tool_checkpoint_start",
        "toolCallId": prepared.tool_call.id,
        "toolName": prepared.tool_call.name,
        "effectiveArgs": prepared.args,
    })
    try:
        # updates 收集所有 on_update 推送的异步任务
        # (emit 可能是 async 函数,所以要收集起来统一 await)
        updates: list[Awaitable[Any] | Any] = []

        def _on_update(partial_result: AgentToolResult) -> None:
            """工具执行中的进度回调,推送 tool_execution_update 事件"""
            updates.append(
                emit(
                    {
                        "type": "tool_execution_update",
                        "toolCallId": prepared.tool_call.id,
                        "toolName": prepared.tool_call.name,
                        "args": prepared.tool_call.arguments,
                        "partialResult": partial_result,
                    }
                )
            )

        # 调用工具的 execute 方法
        # execute 的签名: (tool_call_id, args, signal, on_update) -> AgentToolResult | Awaitable[AgentToolResult]
        raw_result = prepared.tool.execute(prepared.tool_call.id, prepared.args, signal, _on_update)
        # 可能是同步也可能是异步,用 _maybe_await 包一下
        result = await _maybe_await(raw_result)
        throw_if_cancelled(signal)

        # 等待所有 on_update 推送完成
        for u in updates:
            await _maybe_await(u)

        return _ExecutedToolCall(result=result, is_error=False)

    except Exception as exc:
        # 工具执行抛异常,包成错误结果
        return _ExecutedToolCall(result=_error_tool_result(str(exc)), is_error=True)


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCall,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> ToolResultMessage:
    """
    完成工具调用:调用 after_tool_call 钩子,构造 ToolResultMessage。

    after_tool_call 钩子可以修改结果、记录日志、触发后续操作等。

    返回: ToolResultMessage(要发回给模型的消息)
    """
    result = executed.result
    is_error = executed.is_error
    throw_if_cancelled(signal)

    # 调用 after_tool_call 钩子(可选)
    if config.after_tool_call:
        after = await _maybe_await(
            config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=prepared.tool_call,
                    args=prepared.args,
                    result=result,
                    is_error=is_error,
                    context=current_context,
                ),
                signal,
            )
        )
        throw_if_cancelled(signal)
        if after:
            if after.content is not None:
                result.content = after.content
            if after.details is not None:
                result.details = after.details
            if after.is_error is not None:
                is_error = after.is_error

    await _emit(
        emit,
        {
            "type": "tool_execution_end",
            "toolCallId": prepared.tool_call.id,
            "toolName": prepared.tool_call.name,
            "result": result,
            "isError": is_error,
        },
    )

    tool_result_message = ToolResultMessage(
        tool_call_id=prepared.tool_call.id,
        tool_name=prepared.tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=_now_ms(),
    )
    await _emit(emit, {"type": "message_start", "message": tool_result_message})
    await _emit(emit, {"type": "message_end", "message": tool_result_message})
    return tool_result_message


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    """
    串行执行所有工具调用:一个跑完再跑下一个。

    流程(对每个 tool_call):
    1. 推送 tool_execution_start 事件
    2. 调用 _prepare_tool_call(查找工具、审批等)
    3. 如果准备失败,直接推送 tool_execution_end + 错误结果,跳到下一个
    4. 如果准备成功,调用 _execute_prepared_tool_call(真正执行)
    5. 调用 _finalize_executed_tool_call(后处理,构造 ToolResultMessage)
    6. 收集结果

    为什么要串行?因为有些工具有依赖关系,例如"先读文件,再根据内容做事"。
    """
    results: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        throw_if_cancelled(signal)
        # 推送工具执行开始事件
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        # 第一步:准备工具调用
        prepared, immediate, immediate_is_error = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, emit, signal
        )
        if prepared is None:
            # 准备失败(工具不存在、被阻止、审批失败等)
            # 直接推送结束事件 + 错误结果,跳到下一个
            await _emit(
                emit,
                {
                    "type": "tool_execution_end",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "result": immediate,
                    "isError": immediate_is_error,
                },
            )
            # 构造错误的 ToolResultMessage
            msg = ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=immediate.content,
                details=immediate.details,
                is_error=True,
                timestamp=_now_ms(),
            )
            await _emit(emit, {"type": "message_start", "message": msg})
            await _emit(emit, {"type": "message_end", "message": msg})
            results.append(msg)
            continue

        # 第二步:执行工具
        executed = await _execute_prepared_tool_call(prepared, emit, signal)
        # 第三步:后处理,构造 ToolResultMessage
        results.append(
            await _finalize_executed_tool_call(
                current_context, assistant_message, prepared, executed, config, emit, signal
            )
        )
    return results


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
) -> list[ToolResultMessage]:
    """
    并行执行所有工具调用:全部一起跑。

    流程:
    1. 遍历所有 tool_call,逐个准备(prepare)
       - 准备失败的:立刻推送错误结果,加入 immediate_results
       - 准备成功的:加入 prepared_calls 列表
    2. 用 asyncio.gather 并发执行所有准备好的工具
    3. 逐个后处理(finalize),构造 ToolResultMessage
    4. 返回 immediate_results + finalized

    为什么要并行?因为有些工具互不依赖(例如"读三个不同的文件"),并行能节省时间。
    注意:prepare 阶段还是串行的(因为审批流程可能需要用户逐个确认)。
    """
    immediate_results: list[ToolResultMessage] = []  # 准备失败的工具结果
    prepared_calls: list[_PreparedToolCall] = []     # 准备成功的工具调用

    # 第一步:逐个准备工具调用
    for tool_call in tool_calls:
        throw_if_cancelled(signal)
        # 推送工具执行开始事件
        await _emit(
            emit,
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call.id,
                "toolName": tool_call.name,
                "args": tool_call.arguments,
            },
        )
        # 准备工具调用(查找工具、审批等)
        prepared, immediate, immediate_is_error = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, emit, signal
        )
        if prepared is None:
            # 准备失败,推送错误结果
            await _emit(
                emit,
                {
                    "type": "tool_execution_end",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "result": immediate,
                    "isError": immediate_is_error,
                },
            )
            msg = ToolResultMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=immediate.content,
                details=immediate.details,
                is_error=True,
                timestamp=_now_ms(),
            )
            await _emit(emit, {"type": "message_start", "message": msg})
            await _emit(emit, {"type": "message_end", "message": msg})
            immediate_results.append(msg)
        else:
            # 准备成功,加入待执行列表
            prepared_calls.append(prepared)

    # 第二步:并发执行所有准备好的工具
    # asyncio.create_task 创建任务,asyncio.gather 等待所有任务完成
    throw_if_cancelled(signal)
    tasks = [asyncio.create_task(_execute_prepared_tool_call(pc, emit, signal)) for pc in prepared_calls]
    executed_results = await asyncio.gather(*tasks)
    throw_if_cancelled(signal)

    # 第三步:逐个后处理,构造 ToolResultMessage
    finalized: list[ToolResultMessage] = []
    for prepared, executed in zip(prepared_calls, executed_results):
        throw_if_cancelled(signal)
        finalized.append(
            await _finalize_executed_tool_call(
                current_context, assistant_message, prepared, executed, config, emit, signal
            )
        )

    # 返回:先是准备失败的,再是执行完的
    return [*immediate_results, *finalized]
