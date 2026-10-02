"""Conservative recovery: reuse durable results, never guess side effects."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ai.types import AssistantMessage, ToolCall, ToolResultMessage
from .serde import message_from_dict, message_to_dict


def batch_key(message: AssistantMessage) -> str:
    data = json.dumps(message_to_dict(message), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def recover_completed_batch(
    assistant: AssistantMessage,
    results: list[ToolResultMessage],
    journal: list[dict[str, Any]],
) -> list[ToolResultMessage]:
    """Return missing results only when the entire batch is known complete.

    A crash after side effects but before result commit is fundamentally
    ambiguous. Even a read-only flag is not a replay/idempotency contract.
    No context mutation occurs until every invocation has been validated.
    """
    calls = [b for b in assistant.content if isinstance(b, ToolCall)]
    if not calls or len({c.id for c in calls}) != len(calls):
        raise ValueError("无法恢复：工具批次为空或调用 ID 重复")
    known = {r.tool_call_id: r for r in results}
    call_ids = {c.id for c in calls}
    if len(known) != len(results) or any(r.tool_call_id not in call_ids for r in results):
        raise ValueError("无法恢复：工具结果 ID 重复或不属于当前批次")
    key = batch_key(assistant)
    # Use only commits after the latest matching batch intent. This prevents
    # an identical response from a previous attempt from supplying stale
    # results if a provider accidentally reuses call IDs.
    markers = [i for i,e in enumerate(journal)
               if e.get("kind") == "tool_batch_committed" and e.get("payload", {}).get("batch_key") == key]
    committed = journal[markers[-1] + 1:] if markers else []
    for entry in committed:
        payload = entry.get("payload", {})
        if entry.get("kind") == "tool_result_committed" and payload.get("batch_key") == key:
            result = message_from_dict(payload["message"])
            if isinstance(result, ToolResultMessage):
                if result.tool_call_id not in call_ids:
                    raise ValueError("无法恢复：已提交结果不属于当前工具批次")
                old = known.get(result.tool_call_id)
                if old is not None and message_to_dict(old) != message_to_dict(result):
                    raise ValueError("无法恢复：同一调用存在冲突的结果记录")
                known[result.tool_call_id] = result
    missing = [c.id for c in calls if c.id not in known or known[c.id].tool_name != c.name]
    if missing:
        raise ValueError("无法自动恢复：工具结果未可靠提交，无法确认工具是否已经产生副作用；请人工检查。调用 ID: " + ", ".join(missing))
    existing = {r.tool_call_id for r in results}
    return [known[c.id] for c in calls if c.id not in existing]
