from __future__ import annotations

"""coding_agent 默认工具 Hook。

这里放的是“应用层”的默认策略，而不是某个具体工具的实现：

* ``create_tool_audit_hook`` 记录模型请求调用了哪个工具；
* ``create_workspace_path_guard`` 阻止工具通过 ``path``/``cwd`` 逃出工作区；
* ``create_read_only_tool_guard`` 在只读会话中阻止修改性工具；
* ``create_secret_redaction_hook`` 在结果交给模型和持久化前脱敏。

这些 Hook 会由 ``coding_agent.factory.create_agent_session`` 自动安装。
扩展仍然可以注册自己的 Hook；默认 Hook 只是应用层的第一道安全边界。
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from ai.types import TextContent
from agent_core import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)


logger = logging.getLogger("loopweaver.coding_agent.hooks")

_PATH_ARGUMENTS = ("path", "cwd")
_REDACTED = "[REDACTED]"

# 这里只匹配常见的“看起来像密钥”的格式，避免把普通业务文本全部替换。
# 例如 OpenAI/兼容 API key、GitHub token、AWS access key、Slack token 和 Bearer token。
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
)

# details 通常是 JSON 风格的字典。只对明确的凭据字段脱敏，不把 token_count、
# tool_call_id 这类普通诊断字段误判成密钥。
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(?:^|[_-])(api[_-]?key|access[_-]?token|auth(?:orization)?|"
    r"client[_-]?secret|private[_-]?key|password|passphrase|cookie|secret)(?:$|[_-])"
)


def _find_tool(context: BeforeToolCallContext | AfterToolCallContext) -> Any | None:
    """从 Hook 上下文里找到本次调用对应的 AgentTool。

    agent_core 在调用 Hook 前已经完成工具名称查找，因此正常情况下这里一定
    能找到工具。这里仍然做防御性查找，因为扩展开发者可能在测试中手工构造
    ``BeforeToolCallContext``，或者上下文中的工具列表在运行时被替换。
    """

    tools = getattr(context.context, "tools", [])
    return next((tool for tool in tools if tool.name == context.tool_call.name), None)


def _path_text(value: Any) -> str | None:
    """把 JSON 参数中的路径转换为文本。

    模型正常会传字符串。额外接受 ``os.PathLike`` 是为了方便 Python 扩展在
    单元测试或内部调用中传入 ``Path``；其他类型直接返回 None，由路径 Guard
    拒绝，而不是把列表/字典隐式拼成一个不可预测的路径。
    """

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, os.PathLike):
        converted = os.fspath(value)
        if isinstance(converted, str):
            return converted.strip()
    return None


def create_workspace_path_guard(
    workspace_dir: str | Path,
) -> Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None]:
    """创建一个限制工具路径必须位于工作区内的 Before Hook。

    为什么需要单独的 Hook？内置 ``read``/``write``/``bash`` 等工具自身已经
    使用 Workspace Resolver，但扩展工具和 MCP 工具可能没有使用它们。这个
    Hook 在所有工具真正执行前统一检查 ``path`` 与 ``cwd``，因此自定义工具
    也不能通过参数把操作目录指到工作区外。

    ``Path.resolve()`` 还会解析 ``..`` 和符号链接，所以以下情况都会被拦截：

    * ``../outside.txt`` 这样的目录穿越；
    * 指向工作区外部的绝对路径；
    * 工作区内、但最终目标是外部目录的符号链接。

    工作区内的绝对路径是允许的；不存在但位于工作区内的目标也允许，因为
    ``write`` 需要能够创建新文件。
    """

    workspace = Path(workspace_dir).resolve()

    def _guard(context: BeforeToolCallContext, signal: Any | None = None) -> BeforeToolCallResult | None:
        _ = signal
        tool = _find_tool(context)
        if tool is None:
            return None

        # 既看 schema 中声明的字段，也看实际参数中的字段。这样即使扩展的
        # schema 写得不完整，只要它收到了 path/cwd，仍然会得到工作区保护。
        schema = getattr(tool, "parameters", {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        declared = properties.keys() if isinstance(properties, dict) else ()
        args = context.args if isinstance(context.args, dict) else {}
        keys = [key for key in _PATH_ARGUMENTS if key in declared or key in args]

        for key in keys:
            if key not in args or args[key] is None:
                continue
            raw_path = _path_text(args[key])
            if raw_path is None:
                return BeforeToolCallResult(
                    block=True,
                    reason=f"Tool {tool.name} argument {key} must be a path string",
                )
            if not raw_path:
                # 空路径和内置工具的默认值一致，代表当前工作区。
                continue
            try:
                target = (workspace / raw_path).resolve()
                target.relative_to(workspace)
            except (OSError, RuntimeError, ValueError):
                # 不把真实外部路径回显给模型，避免把本机目录结构泄露到对话里。
                return BeforeToolCallResult(
                    block=True,
                    reason=f"Tool {tool.name} argument {key} escapes the workspace boundary",
                )

        return None

    return _guard


def create_read_only_tool_guard(
    enabled: bool,
) -> Callable[[BeforeToolCallContext, Any | None], BeforeToolCallResult | None]:
    """创建只读会话 Guard。

    工厂目前已经会从工具列表中过滤大部分修改性工具；这个 Hook 是第二道
    防线，尤其保护同名扩展覆盖内置工具、或者会话后续动态加入工具的情况。
    工具必须显式声明 ``read_only=True`` 才能在只读会话中执行。
    """

    def _guard(context: BeforeToolCallContext, signal: Any | None = None) -> BeforeToolCallResult | None:
        _ = signal
        if not enabled:
            return None
        tool = _find_tool(context)
        if tool is not None and not bool(getattr(tool, "read_only", False)):
            return BeforeToolCallResult(
                block=True,
                reason=f"Tool {tool.name} is disabled in read-only mode",
            )
        return None

    return _guard


def create_tool_audit_hook(
    audit_logger: logging.Logger | None = None,
) -> Callable[[BeforeToolCallContext, Any | None], None]:
    """创建一个 fail-open 的工具调用审计 Hook。

    审计只记录工具名、调用 ID、只读属性和参数名，不记录参数值。这样日志能
    帮助定位“模型准备调用了什么”，又不会把命令行参数、密码或文件内容直接
    复制进日志。它是观测型 Hook，日志系统自身出错时不应该阻断主任务。
    """

    target_logger = audit_logger or logger

    def _audit(context: BeforeToolCallContext, signal: Any | None = None) -> None:
        _ = signal
        try:
            tool = _find_tool(context)
            target_logger.info(
                "tool_call_requested tool=%s tool_call_id=%s read_only=%s arg_names=%s",
                context.tool_call.name,
                context.tool_call.id,
                bool(getattr(tool, "read_only", False)) if tool is not None else None,
                sorted(str(key) for key in context.args.keys()),
            )
        except Exception:
            # 审计是观测能力，不应因为自定义 logging handler 的异常破坏工具执行。
            return None
        return None

    return _audit


def _redact_text(text: str) -> tuple[str, bool]:
    """替换文本中常见的凭据格式，并返回 ``(文本, 是否发生变化)``。"""

    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: "Bearer " + _REDACTED
            if match.group(0).lower().startswith("bearer ")
            else _REDACTED,
            redacted,
        )
    return redacted, redacted != text


def _redact_value(value: Any) -> tuple[Any, bool]:
    """递归处理 details 中常见的 JSON-like 值。

    对敏感字段直接替换整个值；普通字符串仍会检查常见密钥格式。未变化时
    返回原对象，避免无意义地复制大型诊断结构。
    """

    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, dict):
        changed = False
        output: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_KEY_PATTERN.search(key):
                output[key] = _REDACTED
                changed = True
                continue
            redacted_item, item_changed = _redact_value(item)
            output[key] = redacted_item
            changed = changed or item_changed
        return (output, True) if changed else (value, False)

    if isinstance(value, list):
        output_list: list[Any] = []
        changed = False
        for item in value:
            redacted_item, item_changed = _redact_value(item)
            output_list.append(redacted_item)
            changed = changed or item_changed
        return (output_list, True) if changed else (value, False)

    if isinstance(value, tuple):
        output_tuple: list[Any] = []
        changed = False
        for item in value:
            redacted_item, item_changed = _redact_value(item)
            output_tuple.append(redacted_item)
            changed = changed or item_changed
        return (tuple(output_tuple), True) if changed else (value, False)

    return value, False


def create_secret_redaction_hook() -> Callable[[AfterToolCallContext, Any | None], AfterToolCallResult | None]:
    """创建最终工具结果脱敏 Hook。

    After Hook 执行完后，agent_core 才会构造 ``ToolResultMessage``，随后该消息
    才会发给模型并进入会话历史。因此把这个 Hook 放在 After 链最后，可以覆盖
    内置工具、扩展工具和前面其他 After Hook 返回的文本/详情。
    """

    def _redact(context: AfterToolCallContext, signal: Any | None = None) -> AfterToolCallResult | None:
        _ = signal
        content_changed = False
        redacted_content: list[Any] = []
        for block in context.result.content:
            if isinstance(block, TextContent):
                text, changed = _redact_text(block.text)
                content_changed = content_changed or changed
                if changed:
                    redacted_content.append(
                        TextContent(
                            type=block.type,
                            text=text,
                            text_signature=block.text_signature,
                        )
                    )
                else:
                    redacted_content.append(block)
            else:
                # 图片等非文本内容不做字符串替换，保持原结果不变。
                redacted_content.append(block)

        redacted_details, details_changed = _redact_value(context.result.details)
        if not content_changed and not details_changed:
            return None
        return AfterToolCallResult(
            content=redacted_content if content_changed else None,
            details=redacted_details if details_changed else None,
        )

    return _redact


__all__ = [
    "create_read_only_tool_guard",
    "create_secret_redaction_hook",
    "create_tool_audit_hook",
    "create_workspace_path_guard",
]
