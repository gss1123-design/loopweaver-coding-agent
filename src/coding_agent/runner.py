from __future__ import annotations

"""
运行模式入口。

当前支持：
- print: 单次问答，输出文本与工具事件
- interactive: 交互式 REPL
"""

from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
import asyncio
import inspect
import json
import sys
from typing import Any, Callable

from ai.types import AssistantMessage, TextContent
from agent_core import AgentEvent

from .agent_session import AgentSession
from .command_registry import format_commands_for_help, list_runtime_commands, resolve_registered_command
from .extensions.types import ExtensionCommandContext, SkillSpec
from .tracing import format_trace, format_trace_list
from .types import InputFn, OutputFn, RunMode


@dataclass
class RunOptions:
    mode: RunMode
    session: AgentSession
    prompt: str | None = None
    output: OutputFn = print
    input_fn: InputFn = input
    show_tool_events: bool = True
    exit_commands: tuple[str, ...] = field(default_factory=lambda: ("exit", "quit", ":q"))


def _extract_assistant_text(message: AssistantMessage) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextContent)).strip()


def _query_trace_history(
    session: AgentSession,
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    operation_id: str | None = None,
    trace_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read Trace history through the session query API.

    The small fallback keeps the runner usable with old test doubles and
    third-party session objects that only expose ``last_trace``.
    """

    query = getattr(session, "query_traces", None)
    if callable(query):
        return list(
            query(
                thread_id=thread_id,
                run_id=run_id,
                operation_id=operation_id,
                trace_id=trace_id,
                limit=limit,
            )
        )

    trace = getattr(session, "last_trace", None)
    if not isinstance(trace, dict):
        return []
    if run_id is not None and trace.get("run_id") != run_id:
        return []
    if operation_id is not None and trace.get("operation_id") != operation_id:
        return []
    if trace_id is not None and trace.get("trace_id") != trace_id:
        return []
    return [trace]


async def run_print(
    session: AgentSession,
    prompt: str,
    *,
    output: OutputFn = print,
    show_tool_events: bool = True,
    skill: SkillSpec | None = None,
    resume: bool = False,
) -> AssistantMessage | None:
    """
    单次问答模式：
    - 监听流式 delta
    - 打印工具执行开始/结束
    - 返回最后一条 assistant 消息
    """

    deltas: list[str] = []

    def on_event(event: AgentEvent) -> None:
        t = event["type"]
        if show_tool_events and t in {"tool_execution_start", "tool_execution_end"}:
            output(f"[tool-event] {t}: {event.get('toolName', '')}")
            return

        if t == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = str(assistant_event.get("delta", ""))
                deltas.append(delta)

    unsubscribe = session.subscribe(on_event)
    try:
        if resume:
            await session.resume_run()
        elif skill is not None:
            await session.prompt_with_skill(skill, prompt)
        else:
            await session.prompt(prompt)
    finally:
        unsubscribe()

    final_assistant = next((m for m in reversed(session.messages) if isinstance(m, AssistantMessage)), None)

    if deltas:
        output("".join(deltas).strip())
    elif final_assistant is not None:
        output(_extract_assistant_text(final_assistant) or "(empty)")

    if final_assistant is not None:
        output(f"[assistant.stop_reason] {final_assistant.stop_reason}")
        output(f"[assistant.error_message] {final_assistant.error_message}")
    return final_assistant


async def run_interactive(
    session: AgentSession,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    show_tool_events: bool = True,
    exit_commands: tuple[str, ...] = ("exit", "quit", ":q"),
) -> None:
    """
    交互模式：
    持续读取输入并执行 prompt，直到命中退出命令。
    """

    output("Entering interactive mode. Type 'exit' or '/exit' to quit.")
    output(format_commands_for_help(session))
    recovery_getter = getattr(session,"recovery_status",None)
    if callable(recovery_getter) and session.messages:
        recovery = recovery_getter()
        if recovery.get("action") != "new_task":
            output("[recovery] " + recovery.get("reason","") + "；先用 /recovery 检查，再选择 /resume 或发送明确的新指令。")
    current_session = session
    while True:
        text = input_fn("you> ").strip()
        bare = text.lstrip("/")
        if bare in exit_commands:
            output("Bye.")
            return
        if not text:
            continue
        if text.startswith("/"):
            handled, switched = await _handle_interactive_command(
                current_session,
                text,
                output=output,
                show_tool_events=show_tool_events,
            )
            if switched is not None:
                current_session.close()
                current_session = switched
            if handled:
                continue
        await run_print(current_session, text, output=output, show_tool_events=show_tool_events)


def _create_fresh_session(old: AgentSession) -> AgentSession:
    """创建全新空白 session，保留模型/工具/设置但不带历史消息。"""
    from .session_store import new_session_id
    from .types import AgentSessionOptions

    if old.mcp_client_owned and old.mcp_client is not None:
        retain = getattr(old.mcp_client, "retain", None)
        if callable(retain):
            retain()

    return AgentSession(
        AgentSessionOptions(
            model=old.agent.state.model,
            workspace_dir=old.workspace_dir,
            system_prompt=old.agent.state.system_prompt,
            tools=[
                tool
                for tool in old.agent.state.tools
                if not getattr(tool, "_xingclaw_subagent_tool", False)
                and not getattr(tool, "_xingclaw_skill_tool", False)
            ],
            session_id=new_session_id(),
            messages=[],
            thinking_level=old.agent.state.thinking_level,
            tool_execution=old.tool_execution,
            max_turns=old.max_turns,
            max_tokens=old.max_tokens,
            max_context_messages=old.max_context_messages,
            max_context_tokens=old.max_context_tokens,
            retain_recent_messages=old.retain_recent_messages,
            summary_builder=old.summary_builder,
            retry_enabled=old.retry_enabled,
            max_retries=old.max_retries,
            retry_base_delay_ms=old.retry_base_delay_ms,
            mcp_servers=old.mcp_servers,
            mcp_client=old.mcp_client,
            mcp_client_owned=old.mcp_client_owned,
            extension_commands=old.extension_commands,
            before_prompt_hooks=old.before_prompt_hooks,
            after_prompt_hooks=old.after_prompt_hooks,
            before_tool_call=old.before_tool_call,
            after_tool_call=old.after_tool_call,
            skills=old.skills,
            enable_skill_tool=old.enable_skill_tool,
            enable_subagent_tool=old.enable_subagent_tool,
            subagent_read_only=old.subagent_read_only,
            subagent_timeout_seconds=old.subagent_timeout_seconds,
            subagent_max_output_chars=old.subagent_max_output_chars,
            approval_gate=old.approval_gate,
        )
    )


async def _handle_interactive_command(
    session: AgentSession,
    text: str,
    *,
    output: OutputFn = print,
    show_tool_events: bool = True,
) -> tuple[bool, AgentSession | None]:
    cmd, _, rest = text.partition(" ")
    arg = rest.strip()
    if cmd == "/help":
        output(format_commands_for_help(session))
        return True, None
    if cmd == "/session":
        output(f"session_id={session.session_id} leaf_id={session.get_leaf_id()}")
        return True, None
    if cmd == "/resume":
        if arg:
            output("usage: /resume")
            return True, None
        try:
            await run_print(
                session,
                "",
                output=output,
                show_tool_events=show_tool_events,
                resume=True,
            )
        except ValueError as exc:
            output(str(exc))
        return True, None
    if cmd == "/recovery":
        output("usage: /recovery" if arg else json.dumps(session.recovery_status(),ensure_ascii=False,indent=2))
        return True,None
    if cmd in {"/workers","/worker"}:
        try:
            if cmd == "/workers":
                if arg:
                    raise ValueError("usage: /workers")
                value = session.worker_summaries()
            else:
                if not arg or " " in arg:
                    raise ValueError("usage: /worker <lane_id>")
                value = session.inspect_worker(arg)
            output(json.dumps(value,ensure_ascii=False,indent=2))
        except (ValueError,OSError) as exc:
            output(str(exc))
        return True,None
    if cmd == "/trace":
        output(format_trace(session.last_trace))
        return True, None
    if cmd == "/traces":
        # No argument lists recent history.  An argument first tries the
        # concrete run_id, then the logical operation_id/trace_id shown by
        # older UIs.  This lets a user copy the id from a previous trace line.
        if arg:
            traces: list[dict[str, Any]] = []
            for field in ("run_id", "operation_id", "trace_id"):
                traces = _query_trace_history(session, **{field: arg}, limit=50)
                if traces:
                    break
        else:
            traces = _query_trace_history(session, limit=20)
        output(format_trace_list(traces, thread_id=session.session_id))
        return True, None
    if cmd == "/tree":
        entries = session.list_entries()
        if not entries:
            output("(empty)")
            return True, None
        for item in entries:
            depth = int(item.get("depth", 0))
            prefix = "  " * max(depth, 0)
            leaf_mark = " *" if item.get("is_leaf") else ""
            output(f"{prefix}- {item.get('id')}{leaf_mark}")
        return True, None
    if cmd == "/clear":
        fresh = _create_fresh_session(session)
        output(f"context cleared → new session_id={fresh.session_id}")
        return True, fresh
    if cmd in {"/new", "/fork"}:
        from_entry = arg or session.get_leaf_id() or ""
        if not from_entry:
            output("cannot resolve source entry")
            return True, None
        forked = session.fork_from_entry(from_entry)
        output(f"forked to session_id={forked.session_id}")
        return True, forked
    if cmd == "/switch":
        if not arg:
            output("usage: /switch <entry_id>")
            return True, None
        session.switch_to_entry(arg)
        output(f"switched leaf -> {session.get_leaf_id()}")
        return True, None
    reg = resolve_registered_command(session, cmd)
    if reg:
        value = reg.handler(
            ExtensionCommandContext(
                name=reg.name,
                args=[p for p in arg.split(" ") if p],
                raw_text=text,
                session=session,
                message=None,
            )
        )
        if inspect.isawaitable(value):
            value = await value
        if reg.source == "skill":
            # Skill 命令不是“打印一段说明”这么简单，而是一次真正的
            # Agent 请求。技能正文在本次 prompt 中按需加载；普通 extension
            # command 仍然只输出 handler 返回值。
            if reg.skill is not None:
                skill_task = arg or f"请根据当前会话上下文执行技能「{reg.skill.name}」。"
                await run_print(
                    session,
                    skill_task,
                    output=output,
                    show_tool_events=show_tool_events,
                    skill=reg.skill,
                )
            elif value:
                await run_print(
                    session,
                    str(value),
                    output=output,
                    show_tool_events=show_tool_events,
                )
        elif value:
            output(str(value))
        return True, None
    return False, None


async def run(options: RunOptions) -> AssistantMessage | None:
    """
    统一运行入口。
    """

    if options.mode == "print":
        if not options.prompt:
            raise ValueError("print mode requires prompt")
        return await run_print(
            options.session,
            options.prompt,
            output=options.output,
            show_tool_events=options.show_tool_events,
        )

    if options.mode == "rpc":
        await run_rpc(options.session, output=options.output)
        return None

    await run_interactive(
        options.session,
        input_fn=options.input_fn,
        output=options.output,
        show_tool_events=options.show_tool_events,
        exit_commands=options.exit_commands,
    )
    return None


async def run_rpc(
    session: AgentSession,
    *,
    output: OutputFn = print,
) -> None:
    """
    极简 RPC 模式（jsonl）：
    - {"type":"prompt","text":"..."}
    - {"type":"continue"}
    - {"type":"state"}
    - {"type":"traces", "thread_id":"...", "run_id":"..."}
    - {"type":"shutdown"}
    """

    def _json_default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, set):
            return list(value)
        return str(value)

    def _emit(obj: dict[str, Any]) -> None:
        output(json.dumps(obj, ensure_ascii=False, default=_json_default))

    def _emit_error(*, req_id: Any, command: Any, code: str, message: str) -> None:
        _emit(
            {
                "type": "response",
                "id": req_id,
                "command": command,
                "status": "error",
                "error": {"code": code, "message": message},
            }
        )

    def _emit_ok(*, req_id: Any, command: str, data: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "type": "response",
            "id": req_id,
            "command": command,
            "status": "ok",
        }
        if data is not None:
            payload["data"] = data
        _emit(payload)

    unsubscribe = session.subscribe(
        lambda event: _emit(
            {
                "type": "event",
                "event": event,
            }
        )
    )

    # prompt/continue must be serialized because AgentSession rejects a second
    # prompt while one is already running.  Approval commands deliberately do
    # not use this lock: they must be able to wake a prompt waiting in the gate.
    run_lock = asyncio.Lock()
    pending_tasks: set[asyncio.Task[bool]] = set()

    def _pending_approvals() -> list[dict[str, Any]]:
        getter = getattr(session, "pending_approvals", None)
        if not callable(getter):
            return []
        return list(getter())

    def _resolve_approval(command: str, tool_call_id: str) -> bool:
        method_name = "approve_tool" if command == "approve" else "reject_tool"
        resolver = getattr(session, method_name, None)
        return bool(callable(resolver) and resolver(tool_call_id))

    async def _execute_request(req: dict[str, Any]) -> bool:
        """Execute one parsed request; return True when it is shutdown."""
        cmd = req.get("type")
        req_id = req.get("id")

        try:
            if cmd in {"prompt", "continue"}:
                async with run_lock:
                    if cmd == "prompt":
                        await session.prompt(str(req.get("text", "")))
                    else:
                        await session.continue_run()
                _emit_ok(req_id=req_id, command=cmd)
            elif cmd in {"approve", "reject"}:
                raw_id = req.get("tool_call_id", req.get("toolCallId", ""))
                tool_call_id = str(raw_id or "")
                if not tool_call_id:
                    raise ValueError(f"{cmd} requires tool_call_id")
                resolved = _resolve_approval(cmd, tool_call_id)
                _emit_ok(
                    req_id=req_id,
                    command=cmd,
                    data={"tool_call_id": tool_call_id, "resolved": resolved},
                )
            elif cmd == "approvals":
                _emit_ok(
                    req_id=req_id,
                    command="approvals",
                    data={"pending": _pending_approvals()},
                )
            elif cmd == "workers":
                _emit_ok(req_id=req_id,command=cmd,data={"workers":session.worker_summaries()})
            elif cmd == "recovery":
                _emit_ok(req_id=req_id,command=cmd,data=session.recovery_status())
            elif cmd == "resume":
                async with run_lock:
                    await session.resume_run()
                _emit_ok(req_id=req_id,command=cmd)
            elif cmd == "worker":
                lane_id = req.get("lane_id")
                if not isinstance(lane_id,str):
                    raise ValueError("worker requires lane_id")
                _emit_ok(req_id=req_id,command=cmd,data=session.inspect_worker(lane_id))
            elif cmd == "state":
                _emit_ok(
                    req_id=req_id,
                    command="state",
                    data={
                        "session_id": session.session_id,
                        "message_count": len(session.messages),
                        "entry_ids": session.list_entry_ids(),
                        "leaf_id": session.get_leaf_id(),
                        "pending_approvals": _pending_approvals(),
                    },
                )
            elif cmd == "traces":
                raw_thread_id = req.get("thread_id", req.get("threadId"))
                thread_id = str(raw_thread_id) if raw_thread_id not in (None, "") else session.session_id
                raw_run_id = req.get("run_id", req.get("runId"))
                raw_operation_id = req.get("operation_id", req.get("operationId"))
                raw_trace_id = req.get("trace_id", req.get("traceId"))
                raw_limit = req.get("limit", 20)
                try:
                    limit = max(0, min(int(raw_limit), 1000))
                except (TypeError, ValueError) as exc:
                    raise ValueError("traces.limit must be an integer") from exc
                traces = _query_trace_history(
                    session,
                    thread_id=thread_id,
                    run_id=str(raw_run_id) if raw_run_id not in (None, "") else None,
                    operation_id=str(raw_operation_id) if raw_operation_id not in (None, "") else None,
                    trace_id=str(raw_trace_id) if raw_trace_id not in (None, "") else None,
                    limit=limit,
                )
                _emit_ok(
                    req_id=req_id,
                    command="traces",
                    data={
                        "thread_id": thread_id,
                        "run_id": raw_run_id,
                        "operation_id": raw_operation_id,
                        "trace_id": raw_trace_id,
                        "traces": traces,
                    },
                )
            elif cmd == "trace":
                # Compatibility response for old clients.  New clients should
                # use `traces`, which returns a queryable history instead of
                # silently selecting only the newest record.
                raw_thread_id = req.get("thread_id", req.get("threadId"))
                raw_run_id = req.get("run_id", req.get("runId"))
                raw_operation_id = req.get("operation_id", req.get("operationId"))
                raw_trace_id = req.get("trace_id", req.get("traceId"))
                if any(value not in (None, "") for value in (raw_thread_id, raw_run_id, raw_operation_id, raw_trace_id)):
                    thread_id = str(raw_thread_id) if raw_thread_id not in (None, "") else session.session_id
                    matches = _query_trace_history(
                        session,
                        thread_id=thread_id,
                        run_id=str(raw_run_id) if raw_run_id not in (None, "") else None,
                        operation_id=str(raw_operation_id) if raw_operation_id not in (None, "") else None,
                        trace_id=str(raw_trace_id) if raw_trace_id not in (None, "") else None,
                        limit=1,
                    )
                    selected_trace = matches[0] if matches else None
                else:
                    selected_trace = session.last_trace
                _emit_ok(
                    req_id=req_id,
                    command="trace",
                    data={
                        "session_id": session.session_id,
                        "thread_id": session.session_id,
                        "trace": selected_trace,
                    },
                )
            elif cmd == "list_entries":
                _emit_ok(
                    req_id=req_id,
                    command="list_entries",
                    data={
                        "session_id": session.session_id,
                        "entry_ids": session.list_entry_ids(),
                        "entries": session.list_entries(),
                        "leaf_id": session.get_leaf_id(),
                    },
                )
            elif cmd == "show_tree":
                _emit_ok(
                    req_id=req_id,
                    command="show_tree",
                    data={
                        "session_id": session.session_id,
                        "tree": session.get_session_tree(),
                        "leaf_id": session.get_leaf_id(),
                    },
                )
            elif cmd == "entry_path":
                entry_id = str(req.get("entry_id", ""))
                if not entry_id:
                    raise ValueError("entry_path requires entry_id")
                _emit_ok(
                    req_id=req_id,
                    command="entry_path",
                    data={
                        "session_id": session.session_id,
                        "entry_id": entry_id,
                        "path": session.get_entry_path(entry_id),
                    },
                )
            elif cmd == "fork_entry":
                entry_id = str(req.get("entry_id", ""))
                if not entry_id:
                    raise ValueError("fork_entry requires entry_id")
                forked = session.fork_from_entry(entry_id)
                try:
                    _emit_ok(
                        req_id=req_id,
                        command="fork_entry",
                        data={
                            "from_session_id": session.session_id,
                            "from_entry_id": entry_id,
                            "new_session_id": forked.session_id,
                        },
                    )
                finally:
                    forked.close()
            elif cmd == "switch_entry":
                entry_id = str(req.get("entry_id", ""))
                if not entry_id:
                    raise ValueError("switch_entry requires entry_id")
                session.switch_to_entry(entry_id)
                _emit_ok(
                    req_id=req_id,
                    command="switch_entry",
                    data={
                        "session_id": session.session_id,
                        "entry_id": entry_id,
                        "path": session.get_entry_path(entry_id),
                    },
                )
            elif cmd == "get_commands":
                _emit_ok(
                    req_id=req_id,
                    command="get_commands",
                    data={
                        "session_id": session.session_id,
                        "commands": [
                            {"name": c.name, "description": c.description, "source": c.source}
                            for c in list_runtime_commands(session)
                        ],
                    },
                )
            elif cmd == "shutdown":
                # Unblock any prompt waiting for a human decision before we
                # wait for its task below.
                reject_all = getattr(session, "reject_all_approvals", None)
                if callable(reject_all):
                    reject_all()
                else:
                    for item in _pending_approvals():
                        _resolve_approval("reject", str(item.get("tool_call_id", "")))
                _emit_ok(req_id=req_id, command="shutdown")
                return True
            else:
                _emit_error(req_id=req_id, command=cmd, code="unknown_command", message="Unknown command")
        except Exception as exc:
            _emit_error(req_id=req_id, command=cmd, code="execution_error", message=str(exc))
        return False

    _emit({"type": "rpc_ready", "session_id": session.session_id, "protocol_version": "1.4"})
    try:
        stopping = False
        while not stopping:
            # stdin.readline is blocking.  Move only the read to a worker so
            # already-started prompt tasks can continue processing events.
            raw = await asyncio.to_thread(sys.stdin.readline)
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception as exc:
                _emit_error(req_id=None, command=None, code="invalid_json", message=f"Invalid JSON: {exc}")
                continue
            if not isinstance(req, dict):
                _emit_error(req_id=None, command=None, code="invalid_request", message="Request must be object")
                continue

            cmd = req.get("type")
            if cmd in {"prompt", "continue"}:
                task = asyncio.create_task(_execute_request(req))
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)
            else:
                stopping = await _execute_request(req)
    finally:
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        unsubscribe()
