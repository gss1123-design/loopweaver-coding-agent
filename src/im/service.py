from __future__ import annotations

"""
IMService：平台无关的 IM -> Agent 服务层。

核心设计（对标 pi-mom）：
1) 每频道维护长期 AgentSession（缓存实例），而非每条消息新建；
2) 流式更新：先发占位消息，持续 PATCH 更新内容；
3) 成本统计：每次回复附加 token usage；
4) MEMORY.md：全局 + 频道级记忆注入。
"""

import asyncio
import inspect
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.types import AssistantMessage, TextContent, UserMessage
from coding_agent import ApprovalGate, CreateAgentSessionOptions, create_agent_session
from coding_agent.agent_session import AgentSession
from coding_agent.command_registry import format_commands_for_help, resolve_registered_command
from coding_agent.memory import MEMORY_KINDS, MemoryStore
from coding_agent.session_store import SessionStore
from coding_agent.tracing import format_trace, format_trace_list
from coding_agent.extensions.types import ExtensionCommandContext, SkillSpec

from .memory import load_channel_memory
from .inbox import DurableInbox
from .session_router import SessionRouter
from .types import (
    IMCardButton,
    IMAdapter,
    IMChannelInfo,
    IMIncomingMessage,
    IMOutgoingCard,
    IMOutgoingText,
    IMUserInfo,
)

logger = logging.getLogger("loopweaver.im.service")

_STREAM_UPDATE_INTERVAL = 0.8
_THINKING_PLACEHOLDER = "思考中..."
_APPROVAL_ARGS_MAX_CHARS = 1_200
_APPROVAL_VALUE_MAX_CHARS = 400
_APPROVAL_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
}
_APPROVAL_LARGE_TEXT_KEYS = {"content", "new_text", "old_text", "patch"}


def _redact_approval_text(text: str) -> str:
    value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+", "Bearer <redacted>", text)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "<redacted-api-key>", value)
    value = re.sub(
        r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization)\s*([=:])\s*([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
        value,
    )
    return value


def _sanitize_approval_value(value: Any, *, key: str = "") -> Any:
    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in _APPROVAL_SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_approval_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_approval_value(item) for item in value[:20]] + (
            [f"...<{len(value) - 20} more items>"] if len(value) > 20 else []
        )
    if isinstance(value, str):
        if normalized_key in _APPROVAL_LARGE_TEXT_KEYS and len(value) > 120:
            return f"<{len(value)} characters omitted>"
        redacted = _redact_approval_text(value)
        if len(redacted) > _APPROVAL_VALUE_MAX_CHARS:
            return redacted[:_APPROVAL_VALUE_MAX_CHARS] + "...<truncated>"
        return redacted
    return value


def _format_approval_args(args: dict[str, Any]) -> str:
    safe_args = _sanitize_approval_value(args)
    text = json.dumps(safe_args, ensure_ascii=False, default=str)
    if len(text) > _APPROVAL_ARGS_MAX_CHARS:
        return text[:_APPROVAL_ARGS_MAX_CHARS] + "...<truncated>"
    return text


@dataclass
class IMServiceConfig:
    workspace_dir: str | Path
    provider: str
    model_id: str
    read_only_mode: bool = False
    max_reply_chars: int = 4000
    channel_queue_limit: int = 20
    max_turns: int = 50
    # IM 对话默认不应无限携带历史；可通过 CLI 或代码覆盖。
    max_context_messages: int | None = 24
    max_context_tokens: int | None = 12000
    retain_recent_messages: int = 8
    max_output_tokens: int | None = 2048
    # IM 长对话优先使用本地摘要，避免压缩时额外发起一次 LLM 请求。
    llm_compaction: bool = False
    # 用户名/频道名不是回答必需信息，默认关闭可少两次飞书 API 请求和少量 prompt token。
    enrich_prompt_context: bool = False
    stale_event_seconds: float = 300.0
    use_card_reply: bool = True
    show_cost_in_reply: bool = True
    stream_updates: bool = True
    session_idle_timeout: float = 3600.0
    tool_approval: bool = False
    approval_timeout_seconds: float = 300.0
    tool_backend: str = "local"
    sandbox_image: str = "loopweaver-sandbox:local"
    enable_structured_memory: bool = False
    subagent_read_only: bool = False


@dataclass
class _ChannelState:
    """每频道维护的长期状态。"""
    session: AgentSession
    session_id: str
    last_active: float
    user_cache: dict[str, IMUserInfo]
    channel_info: IMChannelInfo | None = None
    approval_gate: ApprovalGate | None = None


@dataclass
class _PendingMemory:
    kind: str
    key: str
    expires_at: float
    quote: str | None = None


class IMService:
    """平台无关的 IM -> Agent 服务层。"""

    def __init__(self, adapter: IMAdapter, config: IMServiceConfig, router: SessionRouter | None = None) -> None:
        self.adapter = adapter
        self.config = config
        self.router = router or SessionRouter(config.workspace_dir)
        self._processed_ids: set[str] = set()
        self._processed_id_order: deque[str] = deque()
        self._processed_id_limit = 2000
        self._inbox = DurableInbox(config.workspace_dir)
        self._inbox_replayed = False
        self._channel_queues: dict[str, deque[IMIncomingMessage]] = {}
        self._channel_running: set[str] = set()
        self._channel_states: dict[str, _ChannelState] = {}
        # Deliberately volatile: after a restart, unconfirmed quotes are gone.
        self._pending_memories: dict[tuple[str, str], _PendingMemory] = {}
        self._idle_sweep_task: asyncio.Task[None] | None = None

    async def handle_webhook(self, headers: dict[str, str], body: bytes) -> dict:
        parsed = self.adapter.handle_webhook(headers, body)
        for message in parsed.messages:
            await self.handle_incoming_message(message)
        return parsed.ack

    async def handle_incoming_message(self, message: IMIncomingMessage) -> None:
        self._evict_idle_channels()
        self._ensure_idle_sweeper()

        # 把上次进程崩溃时仍处于 accepted/processing 的消息重新放回内存队列。
        # 只在第一次 webhook 到达时恢复，避免每条消息重复扫描磁盘。
        if not self._inbox_replayed:
            self._inbox_replayed = True
            for pending in self._inbox.pending_messages():
                pending_key = self._channel_key(pending)
                pending_queue = self._channel_queues.setdefault(pending_key, deque())
                if pending.message_id and any(item.message_id == pending.message_id for item in pending_queue):
                    continue
                pending_queue.append(pending)

        if self._is_duplicate_message(message.message_id):
            logger.warning("skip duplicate message message_id=%s", message.message_id)
            await self._drain_channel_queue(self._channel_key(message))
            return
        if not self._inbox.accept(message):
            logger.warning("skip durable duplicate message message_id=%s", message.message_id)
            self._mark_processed(message.message_id)
            await self._drain_channel_queue(self._channel_key(message))
            return
        channel_key = self._channel_key(message)

        # Approval replies must remain actionable even if the IM platform
        # delivered them late.  Otherwise a prompt waiting on a gate could
        # never be released.
        if await self._handle_approval_command(message):
            self._mark_processed(message.message_id)
            self._inbox.mark_completed(message.message_id)
            return

        # A message that is old while the channel is already busy is usually a
        # legitimate user message waiting behind a slow model/network call.
        # Queue it instead of silently dropping it.  Only drop stale replayed
        # events when the channel is idle (for example after a restart).
        stale = self._is_stale_event(message)
        if stale and channel_key not in self._channel_running:
            logger.warning(
                "skip stale event message_id=%s created_at=%s",
                message.message_id, message.created_at,
            )
            self._mark_processed(message.message_id)
            self._inbox.mark_dropped(message.message_id, "stale")
            return
        if stale:
            logger.info(
                "queue stale event because channel is busy message_id=%s created_at=%s",
                message.message_id, message.created_at,
            )
        queue = self._channel_queues.setdefault(channel_key, deque())
        if len(queue) >= self.config.channel_queue_limit:
            logger.warning("drop message due to queue limit key=%s", channel_key)
            self._mark_processed(message.message_id)
            self._inbox.mark_dropped(message.message_id, "queue_limit")
            return
        queue.append(message)
        if channel_key in self._channel_running:
            return

        await self._drain_channel_queue(channel_key)

    async def _drain_channel_queue(self, channel_key: str) -> None:
        """串行消费一个频道的 durable/in-memory inbox。"""

        queue = self._channel_queues.get(channel_key)
        if not queue or channel_key in self._channel_running:
            return
        self._channel_running.add(channel_key)
        try:
            while queue:
                current = queue.popleft()
                self._inbox.mark_processing(current.message_id)
                try:
                    await self._handle_single_message(current)
                except BaseException as exc:
                    self._inbox.mark_failed(current.message_id, str(exc))
                    raise
                else:
                    self._inbox.mark_completed(current.message_id)
        finally:
            self._channel_running.discard(channel_key)
            if not queue:
                self._channel_queues.pop(channel_key, None)

    async def _handle_single_message(self, message: IMIncomingMessage) -> None:
        memory_pending = (self._channel_key(message), message.user_id) in self._pending_memories
        logger.info(
            "processing message platform=%s channel=%s user=%s text=%r",
            message.platform, message.channel_id, message.user_id,
            "<memory confirmation input>" if memory_pending else message.text.strip()[:80],
        )
        if await self._handle_memory_message(message):
            self._mark_processed(message.message_id)
            return
        if await self._handle_control_command(message):
            self._mark_processed(message.message_id)
            return

        session, _ = await asyncio.to_thread(self._get_or_create_channel_session, message)

        user_context = await self._build_user_context(message)
        prompt_text = message.text
        if user_context:
            prompt_text = f"{user_context}\n\n{message.text}"

        full_reply = await self._run_agent_reply(session, prompt_text, message)

        self._mark_processed(message.message_id)
        logger.info("reply sent channel=%s chars=%d", message.channel_id, len(full_reply))

    async def _handle_memory_message(self, message: IMIncomingMessage) -> bool:
        """Host-only, opt-in confirmed quote flow; never routes quote to the LLM."""
        text = message.text.strip()
        parts = text.split()
        command = parts[0].lower() if parts else ""
        command = {
            "记住": "/remember",
            "忘记": "/forget",
            "确认记忆": "/confirm",
            "取消记忆": "/cancel",
        }.get(command, command)
        identity = (self._channel_key(message), message.user_id)
        pending = self._pending_memories.get(identity)
        if pending is not None and pending.expires_at < time.time():
            self._pending_memories.pop(identity, None)
            pending = None
        if command not in {"/remember", "/forget", "/confirm", "/cancel"} and pending is None:
            return False

        async def reply(body: str) -> None:
            await self._send_text_async(IMOutgoingText(
                channel_id=message.channel_id, text=body,
                thread_id=message.thread_id, reply_to_message_id=message.message_id,
            ))

        # A channel memory is shared by that channel. Do not let one member of
        # a group save or revoke memories for everyone else through this MVP.
        raw = message.raw if isinstance(message.raw, dict) else {}
        event = raw.get("event") if isinstance(raw.get("event"), dict) else {}
        envelope = event.get("message") if isinstance(event.get("message"), dict) else raw.get("message")
        envelope = envelope if isinstance(envelope, dict) else raw
        if envelope.get("chat_type") == "group":
            self._pending_memories.pop(identity, None)
            await reply("确认记忆命令目前仅支持与机器人私聊，不在群聊中保存或撤销。")
            return True

        if not self.config.enable_structured_memory:
            await reply("结构化记忆未启用；启动飞书服务时加入 --structured-memory。")
            return True

        scope = f"im:{message.platform}:{message.channel_id}"
        if command == "/cancel":
            self._pending_memories.pop(identity, None)
            await reply("已取消，本次内容没有写入记忆。")
            return True
        if command == "/remember":
            if len(parts) != 3 or parts[1] not in MEMORY_KINDS or not re.fullmatch(r"[\w.-]{1,80}", parts[2]):
                await reply("用法：记住 <preference|fact|decision|procedure> <key>\n例如：记住 preference editor")
                return True
            self._pending_memories[identity] = _PendingMemory(parts[1], parts[2], time.time() + 300)
            await reply("请在 5 分钟内单独发送要记住的原文（最多 4000 字）；可用「取消记忆」取消。")
            return True
        if command == "/forget":
            if len(parts) != 2:
                await reply("用法：忘记 <key>")
                return True
            record = await asyncio.to_thread(MemoryStore(self.config.workspace_dir).get, scope, parts[1])
            if record is None:
                await reply("未找到这条记忆。")
                return True
            source = str(record["source"])
            match = re.fullmatch(r"session-entry:([A-Za-z0-9_-]{1,128}):([A-Za-z0-9_-]{1,128})", source)
            if match is None:
                await reply("这不是经用户确认写入的记忆，不能用 /forget 撤销。")
                return True
            try:
                await asyncio.to_thread(
                    MemoryStore(self.config.workspace_dir).revoke_confirmed_user_quote,
                    scope, parts[1], session_id=match[1], entry_id=match[2],
                )
            except ValueError as exc:
                await reply(f"撤销失败：{exc}")
            else:
                await reply(f"已撤销记忆：{parts[1]}")
            return True
        if pending is None:
            if command == "/confirm":
                await reply("当前没有待确认的记忆；请先发送「记住 <kind> <key>」。")
                return True
            return False
        if pending.quote is None:
            if command.startswith("/"):
                await reply("请单独发送原文，或用「取消记忆」取消。")
            elif not text or len(text) > 4000:
                await reply("原文须为 1–4000 字；请重新发送。")
            else:
                pending.quote = text
                pending.expires_at = time.time() + 300
                await reply(f"待保存：\n类型：{pending.kind}\n键：{pending.key}\n原文：{text}\n\n确认请发送「确认记忆」；放弃请发送「取消记忆」。")
            return True
        if command != "/confirm" or len(parts) != 1:
            await reply("请发送「确认记忆」保存这条原文，或「取消记忆」放弃。")
            return True

        # Consume the pending decision before writing. A retry cannot reuse it.
        self._pending_memories.pop(identity, None)
        try:
            session_id = self.router.get_or_create_session_id(
                platform=message.platform, channel_id=message.channel_id, thread_id=message.thread_id,
            )
            witness = SessionStore(self.config.workspace_dir, session_id)
            await asyncio.to_thread(
                witness.ensure_initialized,
                model_id=self.config.model_id, provider=self.config.provider, system_prompt="",
            )
            entry_id = await asyncio.to_thread(witness.append_session_message, UserMessage(content=pending.quote))
            await asyncio.to_thread(
                MemoryStore(self.config.workspace_dir).put_confirmed_user_quote,
                scope, pending.key, pending.quote, kind=pending.kind,
                session_id=session_id, entry_id=entry_id,
            )
        except (OSError, ValueError) as exc:
            logger.warning("confirmed memory write failed key=%s: %s", pending.key, exc)
            await reply("保存失败；没有写入记忆。请检查原文和服务状态后重新发起。")
        else:
            await reply(f"已保存记忆：{pending.key}。可用「忘记 {pending.key}」撤销。")
        return True

    async def _run_agent_reply(
        self,
        session: AgentSession,
        prompt_text: str,
        message: IMIncomingMessage,
        *,
        skill: SkillSpec | None = None,
        resume: bool = False,
    ) -> str:
        """统一执行一次 Agent 请求，并负责 IM 回复的收尾处理。

        普通消息和 /skill 命令都必须经过同一条路径，否则两者会出现
        不同的流式、token 统计和截断行为。
        """
        if self.config.stream_updates:
            reply_text = await self._prompt_with_streaming_fast(
                session,
                prompt_text,
                message,
                skill=skill,
                resume=resume,
            )
        else:
            reply_text = await self._prompt_simple(
                session,
                prompt_text,
                message,
                skill=skill,
                resume=resume,
            )

        if not reply_text:
            reply_text = "(empty)"

        cost_line = self._format_cost(session) if self.config.show_cost_in_reply else ""
        if len(reply_text) > self.config.max_reply_chars:
            reply_text = reply_text[: self.config.max_reply_chars] + "\n...<truncated>..."

        full_reply = f"{reply_text}\n\n{cost_line}" if cost_line else reply_text
        if not self.config.stream_updates:
            await self._send_reply_async(message, full_reply)
        return full_reply

    def _get_or_create_channel_session(self, message: IMIncomingMessage) -> tuple[AgentSession, str]:
        """获取或创建频道级长期 session。"""
        channel_key = self._channel_key(message)
        state = self._channel_states.get(channel_key)

        if state is not None:
            state.last_active = time.time()
            return state.session, state.session_id

        session_id = self.router.get_or_create_session_id(
            platform=message.platform,
            channel_id=message.channel_id,
            thread_id=message.thread_id,
        )

        # factory.py 已经把全局 MEMORY.md 放进 system prompt；这里仅注入频道记忆，
        # 避免全局记忆在 IM 层和工厂层重复拼接。
        from .memory import load_merged_memory
        memory_loader = lambda: load_merged_memory(self.config.workspace_dir, message.channel_id)

        approval_gate = (
            ApprovalGate(
                timeout_seconds=self.config.approval_timeout_seconds,
                state_path=Path(self.config.workspace_dir)
                / ".loopweaver"
                / "im"
                / f"approval_{session_id}.json",
            )
            if self.config.tool_approval
            else None
        )
        session = create_agent_session(
            CreateAgentSessionOptions(
                workspace_dir=self.config.workspace_dir,
                tool_backend=self.config.tool_backend,
                sandbox_image=self.config.sandbox_image,
                enable_structured_memory=self.config.enable_structured_memory,
                subagent_read_only=self.config.subagent_read_only,
                memory_scope=f"im:{message.platform}:{message.channel_id}",
                provider=self.config.provider,
                model_id=self.config.model_id,
                max_turns=self.config.max_turns,
                max_tokens=self.config.max_output_tokens,
                max_context_messages=self.config.max_context_messages,
                max_context_tokens=self.config.max_context_tokens,
                retain_recent_messages=self.config.retain_recent_messages,
                summary_builder=(None if self.config.llm_compaction else AgentSession.fallback_summary),
                session_id=session_id,
                read_only_mode=self.config.read_only_mode,
                memory_loader=memory_loader,
                approval_gate=approval_gate,
            )
        )

        channel_info = None
        if self.config.enrich_prompt_context and hasattr(self.adapter, "get_chat_info"):
            channel_info = self.adapter.get_chat_info(message.channel_id)

        self._channel_states[channel_key] = _ChannelState(
            session=session,
            session_id=session_id,
            last_active=time.time(),
            user_cache={},
            channel_info=channel_info,
            approval_gate=approval_gate,
        )
        logger.info("channel session created key=%s session_id=%s", channel_key, session_id)
        self._evict_idle_channels()
        return session, session_id

    def _ensure_idle_sweeper(self) -> None:
        """在当前 IM 事件循环中启动定期清理任务。"""
        if self._idle_sweep_task is None or self._idle_sweep_task.done():
            self._idle_sweep_task = asyncio.create_task(self._idle_sweep_loop())

    async def _idle_sweep_loop(self) -> None:
        interval = max(60.0, min(self.config.session_idle_timeout / 2, 300.0))
        try:
            while True:
                await asyncio.sleep(interval)
                self._evict_idle_channels()
        except asyncio.CancelledError:
            raise

    def close(self) -> None:
        """关闭 IMService 持有的 session、清理任务和适配器连接。"""
        if self._idle_sweep_task is not None and not self._idle_sweep_task.done():
            self._idle_sweep_task.cancel()
        self._idle_sweep_task = None
        for state in self._channel_states.values():
            if state.approval_gate is not None:
                state.approval_gate.reject_all()
            state.session.close()
        self._channel_states.clear()
        close_adapter = getattr(self.adapter, "close", None)
        if callable(close_adapter):
            close_adapter()

    def _evict_idle_channels(self) -> None:
        """清理超时的频道 session。"""
        now = time.time()
        to_remove = [
            key for key, state in self._channel_states.items()
            if now - state.last_active > self.config.session_idle_timeout
        ]
        for key in to_remove:
            state = self._channel_states.pop(key)
            if state.approval_gate is not None:
                state.approval_gate.reject_all()
            state.session.close()
            logger.info("evicted idle channel session key=%s", key)

    def _invalidate_channel_session(self, message: IMIncomingMessage) -> None:
        """强制清除频道 session（用于 /clear 等命令）。"""
        channel_key = self._channel_key(message)
        state = self._channel_states.pop(channel_key, None)
        if state:
            if state.approval_gate is not None:
                state.approval_gate.reject_all()
            state.session.close()

    async def _build_user_context(self, message: IMIncomingMessage) -> str:
        """构建用户上下文信息，注入到 prompt。"""
        if not self.config.enrich_prompt_context:
            return ""
        parts: list[str] = []

        if hasattr(self.adapter, "get_user_info"):
            user_info = await asyncio.to_thread(self.adapter.get_user_info, message.user_id)
            if user_info and user_info.name:
                parts.append(f"[发送者: {user_info.name}]")

        channel_key = self._channel_key(message)
        state = self._channel_states.get(channel_key)
        if state and state.channel_info and state.channel_info.name:
            parts.append(f"[频道: {state.channel_info.name}]")

        return " ".join(parts)

    async def _send_approval_notice(self, message: IMIncomingMessage, event: dict[str, Any]) -> None:
        tool_call_id = str(event.get("toolCallId") or "")
        tool_name = str(event.get("toolName") or "unknown")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        channel_state = self._channel_states.get(self._channel_key(message))
        if channel_state is not None:
            binder = getattr(channel_state.session, "bind_tool_approval_requester", None)
            if callable(binder):
                binder(tool_call_id, message.user_id)
            elif channel_state.approval_gate is not None:
                channel_state.approval_gate.set_requester(tool_call_id, message.user_id)
        args_text = _format_approval_args(args)
        text = (
            f"需要确认执行工具 `{tool_name}`\n"
            f"参数：`{args_text}`\n"
            f"允许：`/approve {tool_call_id}`\n"
            f"拒绝：`/reject {tool_call_id}`"
        )
        if self.config.use_card_reply and hasattr(self.adapter, "send_card"):
            try:
                await asyncio.to_thread(
                    self.adapter.send_card,
                    IMOutgoingCard(
                        channel_id=message.channel_id,
                        title="LoopWeaver 工具审批",
                        markdown_content=(
                            f"**工具：** `{tool_name}`\n\n"
                            f"**参数：** `{args_text}`\n\n"
                            "请确认是否允许执行。"
                        ),
                        thread_id=message.thread_id,
                        reply_to_message_id=message.message_id,
                        buttons=[
                            IMCardButton(
                                text="允许执行",
                                style="primary",
                                value={
                                    "loopweaver_action": "tool_approval",
                                    "decision": "approve",
                                    "tool_call_id": tool_call_id,
                                    "thread_id": message.thread_id or "",
                                },
                            ),
                            IMCardButton(
                                text="拒绝",
                                style="danger",
                                value={
                                    "loopweaver_action": "tool_approval",
                                    "decision": "reject",
                                    "tool_call_id": tool_call_id,
                                    "thread_id": message.thread_id or "",
                                },
                            ),
                        ],
                    ),
                )
                return
            except Exception as exc:
                logger.warning("approval card failed, fallback to text: %s", exc)
        try:
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=text,
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
        except Exception as exc:
            logger.warning("approval notice failed: %s", exc)

    async def _prompt_simple(
        self,
        session: AgentSession,
        text: str,
        message: IMIncomingMessage | None = None,
        *,
        skill: SkillSpec | None = None,
        resume: bool = False,
    ) -> str:
        """非流式：直接调用 prompt 并提取结果。"""
        async def on_event(event: dict[str, Any]) -> None:
            if message is not None and event.get("type") == "approval_required":
                await self._send_approval_notice(message, event)

        subscribe = getattr(session, "subscribe", None)
        unsubscribe = subscribe(on_event) if callable(subscribe) else (lambda: None)
        try:
            await self._prompt_session(session, text, skill=skill, resume=resume)
            return self._extract_last_assistant_text(session)
        except Exception as exc:
            logger.exception("agent prompt failed: %s", exc)
            return f"[IM bridge error] {exc}"
        finally:
            unsubscribe()

    async def _prompt_with_streaming_legacy(
        self,
        session: AgentSession,
        text: str,
        message: IMIncomingMessage,
        *,
        skill: SkillSpec | None = None,
    ) -> str:
        """流式：先发占位消息，持续 PATCH 更新。"""
        placeholder_id: str | None = None
        last_update_time = 0.0
        accumulated_text = ""
        last_sent_text = ""
        update_in_flight = False

        try:
            placeholder_id = self.adapter.send_card(
                IMOutgoingCard(
                    channel_id=message.channel_id,
                    title="LoopWeaver",
                    markdown_content=_THINKING_PLACEHOLDER,
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
        except Exception as exc:
            logger.warning("failed to send placeholder: %s", exc)

        async def _on_event(event: dict[str, Any]) -> None:
            nonlocal last_update_time, accumulated_text, last_sent_text, update_in_flight
            if event.get("type") != "message_update":
                return
            msg = event.get("message")
            if not isinstance(msg, AssistantMessage):
                return
            text_parts = [b.text for b in msg.content if isinstance(b, TextContent)]
            accumulated_text = "".join(text_parts)

            if not placeholder_id or not hasattr(self.adapter, "update_text"):
                return
            # 已有 PATCH 在飞，直接跳过：否则每个 token 都会排队发一次请求，
            # 242 字的回复能积压出上百次调用。
            if update_in_flight:
                return
            now = time.time()
            if now - last_update_time < _STREAM_UPDATE_INTERVAL:
                return
            preview = accumulated_text[:self.config.max_reply_chars] if accumulated_text else _THINKING_PLACEHOLDER
            # 内容没变就不用发
            if preview == last_sent_text:
                return

            update_in_flight = True
            try:
                # update_text 是同步 HTTP 调用，直接调会阻塞 asyncio 事件循环，
                # 导致 DeepSeek 流式输出在 PATCH 期间卡住。
                # 用 run_in_executor 放到线程池里跑，不阻塞事件循环。
                if await self._update_text_async(placeholder_id, preview):
                    last_sent_text = preview
            finally:
                # 时间戳在 PATCH 完成后才记，保证两次请求之间真的隔了一个间隔
                last_update_time = time.time()
                update_in_flight = False

        unsub = session.subscribe(_on_event)
        try:
            await self._prompt_session(session, text, skill=skill)
            final_text = self._extract_last_assistant_text(session) or accumulated_text
        except Exception as exc:
            logger.exception("agent prompt failed: %s", exc)
            final_text = f"[IM bridge error] {exc}"
        finally:
            unsub()

        if placeholder_id and hasattr(self.adapter, "update_text"):
            cost_line = ""
            if self.config.show_cost_in_reply:
                cost_line = self._format_cost(session)
            full = final_text
            if len(full) > self.config.max_reply_chars:
                full = full[: self.config.max_reply_chars] + "\n...<truncated>..."
            if cost_line:
                full = f"{full}\n\n{cost_line}"
            try:
                # 最终更新也放线程池，避免阻塞
                await self._update_text_async(placeholder_id, full, final=True)
            except Exception as exc:
                logger.warning("final stream update failed: %s", exc)
        return final_text

    async def _prompt_with_streaming_fast(
        self,
        session: AgentSession,
        text: str,
        message: IMIncomingMessage,
        *,
        skill: SkillSpec | None = None,
        resume: bool = False,
    ) -> str:
        """流式回复的非阻塞版本：模型继续生成，PATCH 在后台合并发送。"""
        placeholder_id: str | None = None
        accumulated_text = ""
        last_sent_text = ""
        pending_preview: str | None = None
        update_task: asyncio.Task[None] | None = None
        last_update_at = 0.0

        if hasattr(self.adapter, "send_card"):
            try:
                placeholder_id = await asyncio.to_thread(
                    self.adapter.send_card,
                    IMOutgoingCard(
                        channel_id=message.channel_id,
                        title="LoopWeaver",
                        markdown_content=_THINKING_PLACEHOLDER,
                        thread_id=message.thread_id,
                        reply_to_message_id=message.message_id,
                    ),
                )
            except Exception as exc:
                logger.warning("failed to send placeholder: %s", exc)

        async def flush_updates() -> None:
            nonlocal pending_preview, last_sent_text, last_update_at
            while pending_preview is not None:
                # Events can arrive faster than Feishu's PATCH endpoint.  Do
                # not start a new request until the minimum interval has
                # elapsed; pending_preview is intentionally overwritten by
                # newer content while we wait.
                wait_seconds = _STREAM_UPDATE_INTERVAL - (time.monotonic() - last_update_at)
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                if pending_preview is None:
                    break
                preview = pending_preview
                pending_preview = None
                if not placeholder_id or preview == last_sent_text:
                    continue
                try:
                    updated = await self._update_text_async(placeholder_id, preview)
                    if updated:
                        last_sent_text = preview
                finally:
                    last_update_at = time.monotonic()

        async def on_event(event: dict[str, Any]) -> None:
            nonlocal accumulated_text, pending_preview, update_task
            if event.get("type") == "approval_required":
                await self._send_approval_notice(message, event)
                return
            if event.get("type") != "message_update":
                return
            msg = event.get("message")
            if not isinstance(msg, AssistantMessage):
                return
            accumulated_text = "".join(
                block.text for block in msg.content if isinstance(block, TextContent)
            )
            if not placeholder_id:
                return
            preview = accumulated_text[:self.config.max_reply_chars] or _THINKING_PLACEHOLDER
            if preview == last_sent_text:
                return
            pending_preview = preview
            if update_task is None or update_task.done():
                update_task = asyncio.create_task(flush_updates())

        subscribe = getattr(session, "subscribe", None)
        unsubscribe = subscribe(on_event) if callable(subscribe) else (lambda: None)
        try:
            await self._prompt_session(session, text, skill=skill, resume=resume)
            final_text = self._extract_last_assistant_text(session) or accumulated_text
        except Exception as exc:
            logger.exception("agent prompt failed: %s", exc)
            final_text = f"[IM bridge error] {exc}"
        finally:
            unsubscribe()

        if update_task is not None:
            # 模型已经结束后，不再把期间积压的中间预览逐个 PATCH。
            # 只保留当前正在执行的一个请求，随后直接发送最终内容；否则飞书每次
            # PATCH 若耗时数秒，几十个中间版本会让“回答完成”被拖到几分钟以后。
            pending_preview = None
            await update_task

        cost_line = self._format_cost(session) if self.config.show_cost_in_reply else ""
        full = final_text[:self.config.max_reply_chars]
        if len(final_text) > self.config.max_reply_chars:
            full += "\n...<truncated>..."
        if cost_line:
            full = f"{full}\n\n{cost_line}"

        if placeholder_id:
            try:
                if full != last_sent_text:
                    await self._update_text_async(placeholder_id, full, final=True)
            except Exception as exc:
                logger.warning("final stream update failed: %s", exc)
        else:
            await self._send_reply_async(message, full)
        return final_text

    @staticmethod
    async def _prompt_session(
        session: AgentSession,
        text: str,
        *,
        skill: SkillSpec | None = None,
        resume: bool = False,
    ) -> Any:
        """Run a normal or skill-scoped prompt through one compatibility path."""

        if resume:
            return await session.resume_run()
        if skill is not None:
            prompt_with_skill = getattr(session, "prompt_with_skill", None)
            if callable(prompt_with_skill):
                return await prompt_with_skill(skill, text)
        return await session.prompt(text)

    async def _send_reply_async(self, message: IMIncomingMessage, text: str) -> None:
        """把同步适配器调用放入线程池，避免阻塞 Agent 事件循环。"""
        if self.config.use_card_reply and hasattr(self.adapter, "send_card"):
            try:
                await asyncio.to_thread(
                    self.adapter.send_card,
                    IMOutgoingCard(
                        channel_id=message.channel_id,
                        title="LoopWeaver",
                        markdown_content=text,
                        thread_id=message.thread_id,
                        reply_to_message_id=message.message_id,
                    ),
                )
                return
            except Exception as exc:
                logger.warning("card reply failed, fallback to text: %s", exc)
        await asyncio.to_thread(
            self.adapter.send_text,
            IMOutgoingText(
                channel_id=message.channel_id,
                text=text,
                thread_id=message.thread_id,
                reply_to_message_id=message.message_id,
            ),
        )

    async def _send_text_async(self, message: IMOutgoingText) -> None:
        await asyncio.to_thread(self.adapter.send_text, message)

    async def _update_text_async(self, message_id: str, text: str, *, final: bool = False) -> bool:
        """PATCH a Feishu card with bounded retries for transient failures.

        A permanent 4xx (for example an expired/deleted message) is not
        retried.  Transport failures, 408/429 and 5xx responses get at most
        two short retries, so a broken stream update cannot delay the final
        agent response indefinitely.
        """
        retryable_statuses = {408, 409, 425, 429}
        attempts = 3
        for attempt in range(attempts):
            try:
                await asyncio.to_thread(self.adapter.update_text, message_id, text)
                return True
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                retryable = status is None or status in retryable_statuses or status >= 500
                if not retryable or attempt >= attempts - 1:
                    label = "final stream" if final else "stream"
                    logger.warning("%s update failed: %s", label, exc)
                    return False
                await asyncio.sleep(0.2 * (attempt + 1))
        return False

    async def _handle_approval_command(self, message: IMIncomingMessage) -> bool:
        """Handle approval commands without entering the normal channel queue.

        A normal prompt occupies the channel worker while the agent waits for a
        human decision.  If ``/approve`` were appended to that same queue, the
        worker could never reach it: the approval command would be waiting
        behind the operation that it is supposed to unblock.  Resolve the gate
        directly instead.
        """
        parts = message.text.strip().split()
        if not parts:
            return False

        command = parts[0].lower()
        if command not in {"/approve", "/reject", "/approvals"}:
            return False

        channel_key = self._channel_key(message)
        state = self._channel_states.get(channel_key)
        gate = state.approval_gate if state is not None else None
        if state is not None:
            state.last_active = time.time()

        if command == "/approvals":
            pending = gate.pending() if gate is not None else []
            pending = [
                item
                for item in pending
                if not item.get("requester_id") or item.get("requester_id") == message.user_id
            ]
            if not pending:
                reply = "当前没有待处理的工具审批请求。"
            else:
                lines = ["待处理的工具审批："]
                for item in pending:
                    raw_args = item.get("args")
                    args = _format_approval_args(raw_args if isinstance(raw_args, dict) else {})
                    recovered = "（原任务已中断，不能直接恢复执行）" if item.get("recovered") else ""
                    lines.append(
                        f"- `{item.get('tool_name', 'unknown')}` "
                        f"id=`{item.get('tool_call_id', '')}` 参数=`{args}`{recovered}"
                    )
                reply = "\n".join(lines)
        elif len(parts) != 2:
            reply = f"用法：{command} <tool_call_id>"
        elif gate is None or state is None:
            reply = "当前没有启用工具审批，或没有待处理的审批请求。"
        else:
            tool_call_id = parts[1]
            approved = command == "/approve"
            resolver = getattr(state.session, "resolve_tool_approval", None)
            if callable(resolver):
                outcome = str(
                    resolver(
                        tool_call_id,
                        approved=approved,
                        actor_id=message.user_id,
                    )
                )
            else:
                gate_resolver = getattr(gate, "resolve", None)
                if callable(gate_resolver):
                    outcome = str(
                        gate_resolver(
                            tool_call_id,
                            approved,
                            actor_id=message.user_id,
                        )
                    )
                else:
                    legacy = state.session.approve_tool if approved else state.session.reject_tool
                    outcome = ("approved" if approved else "rejected") if legacy(tool_call_id) else "not_found"

            if outcome in {"approved", "rejected"}:
                action = "允许" if outcome == "approved" else "拒绝"
                reply = f"已{action}工具请求：`{tool_call_id}`"
            elif outcome == "unauthorized":
                reply = "你不是这次任务的发起者，不能处理该工具审批。"
            elif outcome == "stale":
                reply = (
                    "该审批来自已经中断的任务，原工具执行协程已不存在，不能直接恢复。"
                    "请重新发起任务，或从最后一个完整消息继续。"
                )
            elif outcome == "already_resolved":
                reply = f"该工具请求已经处理：`{tool_call_id}`"
            else:
                reply = f"未找到待处理的工具请求：`{tool_call_id}`"

        await self._send_text_async(
            IMOutgoingText(
                channel_id=message.channel_id,
                text=reply,
                thread_id=message.thread_id,
                reply_to_message_id=message.message_id,
            )
        )
        return True

    def _send_reply(self, message: IMIncomingMessage, text: str) -> None:
        """发送最终回复（非流式模式下使用）。"""
        if self.config.use_card_reply and hasattr(self.adapter, "send_card"):
            try:
                self.adapter.send_card(
                    IMOutgoingCard(
                        channel_id=message.channel_id,
                        title="LoopWeaver",
                        markdown_content=text,
                        thread_id=message.thread_id,
                        reply_to_message_id=message.message_id,
                    )
                )
                return
            except Exception as exc:
                logger.warning("card reply failed, fallback to text: %s", exc)

        self.adapter.send_text(
            IMOutgoingText(
                channel_id=message.channel_id,
                text=text,
                thread_id=message.thread_id,
                reply_to_message_id=message.message_id,
            )
        )

    @staticmethod
    def _format_cost(session: AgentSession) -> str:
        usage = getattr(session, "last_usage", None)
        if not usage:
            return ""
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        tokens = int(usage.get("total_tokens", 0) or (input_tokens + output_tokens))
        cost = usage.get("cost", {})
        total_cost = cost.get("total", 0.0)
        if tokens <= 0:
            return ""
        parts = [f"tokens: in {input_tokens} | out {output_tokens} | total {tokens}"]
        if total_cost > 0:
            parts.append(f"cost: ${total_cost:.4f}")
        return f"📊 {' | '.join(parts)}"

    async def _handle_control_command(self, message: IMIncomingMessage) -> bool:
        text = message.text.strip()
        if not text.startswith("/"):
            return False

        if text in {"/clear", "/new"}:
            for identity in list(self._pending_memories):
                if identity[0] == self._channel_key(message):
                    self._pending_memories.pop(identity, None)
            self._invalidate_channel_session(message)
            new_session_id = self.router.rotate_session_id(
                platform=message.platform,
                channel_id=message.channel_id,
                thread_id=message.thread_id,
            )
            logger.info("session rotated by command=%s new_session_id=%s", text, new_session_id)
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=f"已新建会话：`{new_session_id}`。后续对话将使用新上下文。",
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        if text == "/session":
            session_id = self.router.get_or_create_session_id(
                platform=message.platform,
                channel_id=message.channel_id,
                thread_id=message.thread_id,
            )
            cum = ""
            channel_key = self._channel_key(message)
            state = self._channel_states.get(channel_key)
            if state:
                usage = state.session.cumulative_usage
                cum = (
                    f"\ntokens: in {usage['input_tokens']} | out {usage['output_tokens']} "
                    f"| total {usage['total_tokens']} | cost: ${usage['total_cost']:.4f}"
                )
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=f"当前会话：`{session_id}`{cum}",
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        if text == "/help":
            session, _ = await asyncio.to_thread(self._get_or_create_channel_session, message)
            help_text = format_commands_for_help(session)
            if self.config.enable_structured_memory:
                help_text += (
                    "\n\n确认记忆：\n"
                    "- `记住 <preference|fact|decision|procedure> <key>` 发起保存；随后单独发送原文，再发「确认记忆」\n"
                    "- `取消记忆` 放弃未确认内容\n"
                    "- `忘记 <key>` 撤销已确认记忆"
                )
            if self.config.tool_approval:
                help_text += (
                    "\n\n工具审批：\n"
                    "- `/approvals` 查看待审批工具\n"
                    "- `/approve <tool_call_id>` 允许执行\n"
                    "- `/reject <tool_call_id>` 拒绝执行"
                )
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=help_text,
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        if text == "/recovery":
            session,_ = await asyncio.to_thread(self._get_or_create_channel_session,message)
            status = await asyncio.to_thread(session.recovery_status)
            await self.adapter.send_reply(IMReply(channel_id=message.channel_id,
                text=json.dumps(status,ensure_ascii=False,indent=2)[:self.max_reply_chars],
                thread_id=message.thread_id,reply_to_message_id=message.message_id))
            return True

        if text == "/resume":
            session, _ = await asyncio.to_thread(self._get_or_create_channel_session, message)
            await self._run_agent_reply(session, "", message, resume=True)
            return True

        # These are built-in commands, not extension commands.  They must be
        # handled here before resolve_registered_command(); that resolver only
        # searches session.extension_commands.  Otherwise /tree, /trace and
        # /traces fall through to _handle_single_message() and are sent to the
        # model as ordinary prompt text.
        parts = text.split()
        command = parts[0].lower() if parts else ""
        if command in {"/workers","/worker"}:
            session,_ = await asyncio.to_thread(self._get_or_create_channel_session,message)
            try:
                if command == "/workers" and len(parts) == 1:
                    value = await asyncio.to_thread(session.worker_summaries)
                elif command == "/worker" and len(parts) == 2:
                    value = await asyncio.to_thread(session.inspect_worker,parts[1])
                else:
                    raise ValueError("用法：/workers 或 /worker <lane_id>")
                reply = json.dumps(value,ensure_ascii=False,indent=2)
                if len(reply) > self.config.max_reply_chars:
                    reply = reply[:self.config.max_reply_chars] + "\n...<truncated>..."
            except (ValueError,OSError) as exc:
                reply = str(exc)
            await self._send_text_async(IMOutgoingText(channel_id=message.channel_id,text=reply,
                thread_id=message.thread_id,reply_to_message_id=message.message_id))
            return True
        if command == "/tree":
            if len(parts) != 1:
                reply = "用法：/tree"
            else:
                session, _ = await asyncio.to_thread(self._get_or_create_channel_session, message)
                entries = session.list_entries()
                if not entries:
                    reply = "(empty)"
                else:
                    lines = [
                        f"session_id={session.session_id}",
                        f"leaf_id={session.get_leaf_id()}",
                    ]
                    for item in entries:
                        try:
                            depth = max(int(item.get("depth", 0)), 0)
                        except (TypeError, ValueError):
                            depth = 0
                        prefix = "  " * depth
                        leaf_mark = " *" if item.get("is_leaf") else ""
                        lines.append(f"{prefix}- {item.get('id')}{leaf_mark}")
                    reply = "\n".join(lines)
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=reply,
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        if command == "/trace":
            if len(parts) != 1:
                reply = "用法：/trace"
            else:
                session, _ = await asyncio.to_thread(self._get_or_create_channel_session, message)
                reply = format_trace(session.last_trace)
                if len(reply) > self.config.max_reply_chars:
                    reply = reply[: self.config.max_reply_chars] + "\n...<truncated>..."
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=reply,
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        if command == "/traces":
            if len(parts) > 2:
                reply = "用法：/traces [run_id|operation_id]"
            else:
                session, _ = await asyncio.to_thread(self._get_or_create_channel_session, message)
                identifier = parts[1] if len(parts) == 2 else None
                if identifier:
                    traces = session.query_traces(run_id=identifier, limit=50)
                    if not traces:
                        traces = session.query_traces(operation_id=identifier, limit=50)
                    if not traces:
                        traces = session.query_traces(trace_id=identifier, limit=50)
                else:
                    traces = session.query_traces(limit=20)

                if identifier and len(traces) == 1:
                    reply = format_trace(traces[0])
                else:
                    reply = format_trace_list(traces, thread_id=session.session_id)
                if len(reply) > self.config.max_reply_chars:
                    reply = reply[: self.config.max_reply_chars] + "\n...<truncated>..."
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=reply,
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
            return True

        parts = text.lstrip("/").split()
        if not parts:
            return False
        cmd_name = parts[0]
        cmd_args = parts[1:]
        session, _ = await asyncio.to_thread(self._get_or_create_channel_session, message)
        cmd = resolve_registered_command(session, cmd_name)
        if not cmd:
            return False
        result = cmd.handler(
            ExtensionCommandContext(
                name=cmd_name,
                args=cmd_args,
                raw_text=text,
                session=session,
                message=message,
            )
        )
        if inspect.isawaitable(result):
            result = await result
        if getattr(cmd, "source", "extension") == "skill":
            skill_value = getattr(cmd, "skill", None)
            skill: SkillSpec | None = skill_value if isinstance(skill_value, SkillSpec) else None
            if skill is not None:
                skill_prompt = " ".join(cmd_args).strip() or f"请根据当前会话上下文执行技能「{skill.name}」。"
            else:
                # Backward compatibility for custom/fake skill commands that
                # predate the lazy-loading SkillSpec field.
                skill_prompt = str(result or text)
            user_context = await self._build_user_context(message)
            if user_context:
                skill_prompt = f"{user_context}\n\n{skill_prompt}"
            full_reply = await self._run_agent_reply(session, skill_prompt, message, skill=skill)
            logger.info(
                "skill command executed command=/%s channel=%s chars=%d",
                cmd_name,
                message.channel_id,
                len(full_reply),
            )
        elif result:
            await self._send_text_async(
                IMOutgoingText(
                    channel_id=message.channel_id,
                    text=str(result),
                    thread_id=message.thread_id,
                    reply_to_message_id=message.message_id,
                )
            )
        return True

    def _is_stale_event(self, message: IMIncomingMessage) -> bool:
        if message.created_at is None:
            return False
        age = time.time() - message.created_at
        threshold = self.config.stale_event_seconds
        return threshold > 0 and age > threshold

    def _is_duplicate_message(self, message_id: str | None) -> bool:
        if not message_id:
            return False
        return message_id in self._processed_ids

    def _mark_processed(self, message_id: str | None) -> None:
        if not message_id:
            return
        if message_id in self._processed_ids:
            return
        self._processed_ids.add(message_id)
        self._processed_id_order.append(message_id)
        while len(self._processed_id_order) > self._processed_id_limit:
            old = self._processed_id_order.popleft()
            self._processed_ids.discard(old)

    @staticmethod
    def _extract_last_assistant_text(session: AgentSession) -> str:
        final_assistant = next(
            (m for m in reversed(session.messages) if isinstance(m, AssistantMessage)),
            None,
        )
        if final_assistant is None:
            return ""
        return "".join(
            block.text for block in final_assistant.content if isinstance(block, TextContent)
        ).strip()

    @staticmethod
    def _channel_key(message: IMIncomingMessage) -> str:
        thread = message.thread_id or "_"
        return f"{message.platform}:{message.channel_id}:{thread}"
