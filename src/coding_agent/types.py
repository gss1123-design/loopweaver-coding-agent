from __future__ import annotations

"""
coding_agent 对外类型定义。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from ai.models import get_model
from ai.types import Message, Model
from agent_core import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentMessage,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionMode,
)
from .extensions.types import LifecycleHook, RegisteredCommand, SkillSpec

ConvertToLlmFn = Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]


@dataclass
class AgentSessionOptions:
    """
    AgentSession 初始化参数。
    """

    model: Model
    workspace_dir: str | Path
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    max_turns: int = 50
    max_tokens: Optional[int] = None
    convert_to_llm: Optional[ConvertToLlmFn] = None
    max_context_messages: Optional[int] = None
    max_context_tokens: Optional[int] = None
    retain_recent_messages: int = 24
    summary_builder: Optional[Callable[[list[Message]], str]] = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    read_only_mode: bool = False
    block_dangerous_bash: bool = True
    bash_allow_patterns: Optional[list[str]] = None
    bash_block_patterns: Optional[list[str]] = None
    edit_require_unique_match: bool = True
    prompt_guidelines: Optional[list[str]] = None
    append_system_prompt: Optional[str] = None
    tool_snippets: Optional[dict[str, str]] = None
    extension_paths: Optional[list[str]] = None
    skill_paths: Optional[list[str]] = None
    prompt_debug_sources: bool = False
    mcp_servers: Optional[list[dict[str, Any]]] = None
    mcp_client: Any | None = None
    mcp_client_owned: bool = False
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    approval_gate: Any | None = None
    # Skill metadata is loaded during session construction; skill bodies are
    # read only when the user or model activates one.
    skills: list[SkillSpec] = field(default_factory=list)
    enable_skill_tool: bool = True
    # Enable the model-visible one-shot ``run_subagent`` tool.  Lanes created
    # internally for delegation disable this flag to prevent recursion.
    enable_subagent_tool: bool = True
    subagent_read_only: bool = False
    subagent_timeout_seconds: float = 120.0
    subagent_max_output_chars: int = 8_000
    memory_loader: Optional[Callable[[], str]] = None
    memory_retrieval_policy: Literal["selective", "legacy"] = "selective"
    tool_backend: Literal["local", "docker"] = "local"
    sandbox_image: str = "xingclaw-sandbox:local"


@dataclass
class CreateAgentSessionOptions:
    """
    更友好的会话创建参数：

    你可以二选一提供模型信息：
    1) 直接传 model；
    2) 传 provider + model_id（由工厂自动解析）。

    若传入已有 session_id，工厂会优先尝试从会话元数据恢复
    provider/model_id/system_prompt。
    """

    workspace_dir: str | Path
    model: Optional[Model] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: str = ""
    tools: list[AgentTool] = field(default_factory=list)
    session_id: Optional[str] = None
    messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: str = "off"
    tool_execution: ToolExecutionMode = "parallel"
    max_turns: int = 50
    max_tokens: Optional[int] = None
    load_workspace_resources: bool = True
    enabled_builtin_tools: Optional[list[str]] = None
    max_context_messages: Optional[int] = None
    max_context_tokens: Optional[int] = None
    retain_recent_messages: int = 24
    summary_builder: Optional[Callable[[list[Message]], str]] = None
    retry_enabled: bool = True
    max_retries: int = 2
    retry_base_delay_ms: int = 1200
    read_only_mode: bool = False
    block_dangerous_bash: bool = True
    bash_allow_patterns: Optional[list[str]] = None
    bash_block_patterns: Optional[list[str]] = None
    edit_require_unique_match: bool = True
    prompt_guidelines: Optional[list[str]] = None
    append_system_prompt: Optional[str] = None
    tool_snippets: Optional[dict[str, str]] = None
    extension_paths: Optional[list[str]] = None
    skill_paths: Optional[list[str]] = None
    prompt_debug_sources: bool = False
    mcp_servers: Optional[list[dict[str, Any]]] = None
    mcp_client: Any | None = None
    mcp_client_owned: bool = False
    extension_commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    before_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    after_prompt_hooks: list[LifecycleHook] = field(default_factory=list)
    before_tool_call: Optional[
        Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None]]
    ] = None
    after_tool_call: Optional[
        Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None | Awaitable[AfterToolCallResult | None]]
    ] = None
    approval_gate: Any | None = None
    enable_skill_tool: bool = True
    enable_subagent_tool: bool = True
    subagent_read_only: bool = False
    subagent_timeout_seconds: float = 120.0
    subagent_max_output_chars: int = 8_000
    tool_backend: Literal["local", "docker"] = "local"
    sandbox_image: str = "xingclaw-sandbox:local"
    enable_structured_memory: bool = False
    memory_retrieval_policy: Literal["selective", "legacy"] = "selective"
    memory_scope: str = "workspace"
    memory_loader: Optional[Callable[[], str]] = None

    def resolve_model(self) -> Model:
        if self.model is not None:
            return self.model
        if self.provider and self.model_id:
            return get_model(self.provider, self.model_id)
        raise ValueError("Model is required: provide model or provider+model_id")


RunMode = Literal["print", "interactive", "rpc"]
OutputFn = Callable[[str], None]
InputFn = Callable[[str], str]
