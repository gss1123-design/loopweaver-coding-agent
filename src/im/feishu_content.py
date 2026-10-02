from __future__ import annotations

"""Turn Feishu text and post message bodies into plain user text."""

import json
from typing import Any


def extract_feishu_text(message_type: str, raw_content: Any) -> str:
    if message_type not in {"text", "post"} or not isinstance(raw_content, str):
        return ""
    try:
        content = json.loads(raw_content)
    except (TypeError, ValueError):
        return raw_content.strip() if message_type == "text" else ""
    if not isinstance(content, dict):
        return ""
    if message_type == "text":
        value = content.get("text")
        return value.strip() if isinstance(value, str) else ""

    # Incoming post content can be a locale map directly or wrapped in `post`.
    post = content.get("post", content)
    if not isinstance(post, dict):
        return ""
    # Incoming events and GET /messages can use the locale-less shape
    # {"title": ..., "content": [[{"tag": "text", ...}]]}.
    locale = post if isinstance(post.get("content"), list) else next(
        (post[name] for name in ("zh_cn", "zh-CN", "en_us", "en-US")
         if isinstance(post.get(name), dict)),
        None,
    )
    if locale is None:
        locale = next((item for item in post.values()
                       if isinstance(item, dict) and isinstance(item.get("content"), list)), None)
    if not isinstance(locale, dict):
        return ""
    lines: list[str] = []
    title = locale.get("title")
    if isinstance(title, str) and title.strip():
        lines.append(title.strip())
    rows = locale.get("content")
    if not isinstance(rows, list):
        return "\n".join(lines)
    for row in rows[:200]:
        if not isinstance(row, list):
            continue
        parts: list[str] = []
        for element in row[:200]:
            if not isinstance(element, dict) or element.get("tag") not in {"text", "a", "code"}:
                continue
            value = element.get("text")
            if isinstance(value, str):
                parts.append(value)
        lines.append("".join(parts))
    return "\n".join(lines).strip()
