from __future__ import annotations

"""Provider-side cancellation compatibility helpers."""

import asyncio
from typing import Any


def throw_if_cancelled(signal: Any | None) -> None:
    """接受 agent_core token 或外部的轻量 signal。"""

    if signal is None:
        return
    throw = getattr(signal, "throw_if_cancelled", None)
    if callable(throw):
        throw()
        return

    value = getattr(signal, "is_cancelled", False)
    if callable(value):
        value = value()
    if not value:
        value = getattr(signal, "cancelled", False)
        if callable(value):
            value = value()
    if not value:
        is_set = getattr(signal, "is_set", None)
        value = is_set() if callable(is_set) else False
    if value:
        raise asyncio.CancelledError
