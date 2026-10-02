from __future__ import annotations

"""
agent_core 的类型定义。

这一层关注“编排”而不是“具体 provider 实现”：
1) 维护 Agent 状态；
2) 定义工具执行协议；
3) 定义循环配置和事件格式。
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, Protocol, TypedDict

from ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
)


ToolExecutionMode = Literal["sequential", "parallel"]

# 当前阶段只支持 LLM 消息类型，后续可以扩展 custom message。
AgentMessage = Message


@dataclass
class AgentToolResult:
    """工具执行产物：内容块 + 细节对象（UI/日志可用）。"""

    content: list[TextContent | ImageContent]
    details: Any = None


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class ToolExecuteFn(Protocol):
    def __call__(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> Awaitable[AgentToolResult] | AgentToolResult: ...


@dataclass
class AgentTool:
    """
    Agent 可执行工具定义。

    注意：name/description/parameters 同时也会作为 LLM 可见的工具元信息。
    """

    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execute: ToolExecuteFn
    # 只有明确声明为只读的工具，才允许自动并行。默认 False 是保守策略：
    # 自定义工具和 MCP 工具如果没有声明，就按可能修改外部状态处理。
    read_only: bool = False
    # 审批是另一条独立的安全策略。默认 True 是 fail-closed：自定义工具
    # 如果没有明确声明“无需人工确认”，在启用 approval_gate 时仍会等待审批。
    requires_approval: bool = True


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] = field(default_factory=list)


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: Optional[str] = None


@dataclass
class AfterToolCallResult:
    content: Optional[list[TextContent | ImageContent]] = None
    details: Any = None
    is_error: Optional[bool] = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class AgentLoopConfig:
    model: Model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
    transform_context: Optional[
        Callable[[list[AgentMessage], Any | None], list[AgentMessage] | Awaitable[list[AgentMessage]]]
    ] = None
    get_api_key: Optional[Callable[[str], str | None | Awaitable[str | None]]] = None
    get_steering_messages: Optional[Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]]] = None
    get_follow_up_messages: Optional[Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]]] = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    approval_gate: Any | None = None
    reasoning: Optional[ThinkingLevel] = None
    session_id: Optional[str] = None
    max_turns: int = 50
    # 单次模型请求的最大输出 token；None 表示交给 provider/model 默认值。
    max_tokens: int | None = None
    # Stable logical operation identity shared by all retry attempts.  These
    # fields stay at the end so positional construction of older configs keeps
    # its original meaning.
    operation_id: Optional[str] = None
    attempt: int = 1


@dataclass
class AgentState:
    system_prompt: str
    model: Model
    thinking_level: Literal["off", "minimal", "low", "medium", "high", "xhigh"] = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


class AgentEventBase(TypedDict):
    """
    所有事件统一元字段（贴近 TS 版 runtime 事件结构）。
    """

    type: str
    runId: str
    turnId: int
    eventId: str
    timestamp: int
    sessionId: str | None


class AIRequestStartEvent(AgentEventBase):
    """开始一次 provider 请求（用于 trace，不落入会话上下文）。"""

    type: Literal["ai_request_start"]
    requestId: str
    provider: str
    model: str
    api: str
    streaming: bool


class AIRequestEndEvent(AgentEventBase):
    """provider 请求结束或失败（用于 trace，不落入会话上下文）。"""

    type: Literal["ai_request_end"]
    requestId: str
    provider: str
    model: str
    api: str
    streaming: bool
    durationMs: int
    chunkCount: int
    timeToFirstChunkMs: int | None
    isError: bool
    errorType: str | None
    responseId: str | None
    stopReason: str | None
    message: AssistantMessage | None


class AgentStartEvent(AgentEventBase):
    type: Literal["agent_start"]


class AgentEndEvent(AgentEventBase):
    type: Literal["agent_end"]
    messages: list[AgentMessage]


class TurnStartEvent(AgentEventBase):
    type: Literal["turn_start"]


class TurnEndEvent(AgentEventBase):
    type: Literal["turn_end"]
    message: AssistantMessage
    toolResults: list[ToolResultMessage]


class MessageStartEvent(AgentEventBase):
    type: Literal["message_start"]
    message: AgentMessage


class MessageUpdateEvent(AgentEventBase):
    type: Literal["message_update"]
    message: AgentMessage
    assistantMessageEvent: dict[str, Any]


class MessageEndEvent(AgentEventBase):
    type: Literal["message_end"]
    message: AgentMessage


class ToolExecutionStartEvent(AgentEventBase):
    type: Literal["tool_execution_start"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]


class ToolExecutionUpdateEvent(AgentEventBase):
    type: Literal["tool_execution_update"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]
    partialResult: AgentToolResult


class ToolExecutionEndEvent(AgentEventBase):
    type: Literal["tool_execution_end"]
    toolCallId: str
    toolName: str
    result: AgentToolResult
    isError: bool


class ApprovalRequiredEvent(AgentEventBase):
    type: Literal["approval_required"]
    toolCallId: str
    toolName: str
    args: dict[str, Any]


class ErrorEvent(AgentEventBase):
    type: Literal["error"]
    error: str


class MaxTurnsReachedEvent(AgentEventBase):
    type: Literal["max_turns_reached"]
    turns: int


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ApprovalRequiredEvent
    | ErrorEvent
    | MaxTurnsReachedEvent
    | AIRequestStartEvent
    | AIRequestEndEvent
)

AgentEventSink = Callable[[AgentEvent], None | Awaitable[None]]
