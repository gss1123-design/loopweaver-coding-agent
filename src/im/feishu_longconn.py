from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from .feishu_content import extract_feishu_text
from .service import IMService
from .types import IMIncomingMessage

logger = logging.getLogger("xingclaw.im.feishu.longconn")


@dataclass
class FeishuLongConnOptions:
    app_id: str
    app_secret: str
    log_level: str = "info"


class _AsyncLoopRunner:
    """给飞书同步回调提供一个长期运行的 asyncio 事件循环。

    lark-oapi 的消息回调是同步函数。如果在回调里直接 asyncio.run，
    回调线程会一直等到模型回答结束，进而拖住 SDK 的心跳和重连逻辑。
    这里让回调只提交任务就立即返回。
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="xingclaw-im-async", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def submit(self, coro: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            return
        future = asyncio.run_coroutine_threadsafe(coro, loop)

        def _done(done_future: Any) -> None:
            try:
                done_future.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("longconn async message task failed")

        future.add_done_callback(_done)

    def close(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def run_feishu_long_connection(service: IMService, options: FeishuLongConnOptions) -> None:
    """使用飞书 SDK 长连接模式收事件。"""

    lark = _import_lark_sdk()
    _warn = getattr(lark.LogLevel, "WARN", None) or getattr(lark.LogLevel, "WARNING", None) or lark.LogLevel.INFO
    log_level_map = {
        "debug": lark.LogLevel.DEBUG,
        "info": lark.LogLevel.INFO,
        "warning": _warn,
        "error": lark.LogLevel.ERROR,
    }
    async_runner = _AsyncLoopRunner()

    def _on_p2_im_message_receive_v1(data: Any) -> None:
        """lark-oapi EventDispatcher 回调，data 是 P2ImMessageReceiveV1 对象。"""
        try:
            raw = _to_dict(data)
            logger.debug("longconn received event payload keys=%s", list(raw.keys()) if raw else "empty")
            msg = _parse_event_object(data) or _parse_ws_message(raw)
            if msg is None:
                logger.debug("longconn event ignored (not a valid text message)")
                return
            logger.info(
                "longconn dispatching message chat_id=%s user=%s text=%r",
                msg.channel_id, msg.user_id, msg.text[:80],
            )
            # 不等待模型完成，立即把控制权还给 lark-oapi 的 WebSocket 回调。
            async_runner.submit(service.handle_incoming_message(msg))
        except Exception as exc:
            logger.exception("failed to process long connection event: %s", exc)

    def _on_p2_card_action_trigger(data: Any) -> Any:
        """Receive an approval-card click and submit it without blocking WS."""

        try:
            msg = _parse_card_action_object(data)
            if msg is not None:
                logger.info(
                    "longconn dispatching card action chat_id=%s user=%s command=%s",
                    msg.channel_id,
                    msg.user_id,
                    msg.text.split(" ", 1)[0],
                )
                async_runner.submit(service.handle_incoming_message(msg))
        except Exception as exc:
            logger.exception("failed to process card action: %s", exc)

        # The callback must return immediately; the actual authorization and
        # gate resolution happen asynchronously and are acknowledged by a
        # normal IM reply.  A toast tells the clicker that the event arrived.
        from lark_oapi.event.callback.model.p2_card_action_trigger import (  # type: ignore
            P2CardActionTriggerResponse,
        )

        return P2CardActionTriggerResponse(
            {"toast": {"type": "info", "content": "审批请求已提交"}}
        )

    logger.info("starting feishu long connection")

    event_handler = (
        lark.EventDispatcherHandler
        .builder("", "")
        .register_p2_im_message_receive_v1(_on_p2_im_message_receive_v1)
        .register_p2_card_action_trigger(_on_p2_card_action_trigger)
        .build()
    )

    client = lark.ws.Client(
        options.app_id,
        options.app_secret,
        event_handler=event_handler,
        log_level=log_level_map.get(options.log_level.lower(), lark.LogLevel.INFO),
    )
    try:
        client.start()
    finally:
        async_runner.close()


def _run_async(coro: Any) -> None:
    """在已有事件循环中调度协程，没有则新建。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import threading

        result_event = threading.Event()
        exception_holder: list[BaseException] = []

        def _thread_target() -> None:
            try:
                asyncio.run(coro)
            except Exception as exc:
                exception_holder.append(exc)
            finally:
                result_event.set()

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()
        result_event.wait(timeout=120)
        if exception_holder:
            logger.exception("async task failed: %s", exception_holder[0])
    else:
        asyncio.run(coro)


def _parse_event_object(data: Any) -> IMIncomingMessage | None:
    """尝试从 lark-oapi 的强类型事件对象中提取消息。"""
    event = getattr(data, "event", None)
    if event is None:
        return None

    message = getattr(event, "message", None)
    sender = getattr(event, "sender", None)
    if message is None or sender is None:
        return None

    sender_type = getattr(sender, "sender_type", "")
    if isinstance(sender_type, str) and sender_type.lower() in {"app", "bot"}:
        return None

    msg_type = getattr(message, "message_type", "")
    if msg_type not in {"text", "post"}:
        logger.info("ignored non-text message_type=%s", msg_type)
        return None

    content_text = extract_feishu_text(msg_type, getattr(message, "content", ""))

    mentions = getattr(message, "mentions", None) or []
    content_text = _strip_bot_mentions_from_objects(content_text, mentions)

    chat_id = getattr(message, "chat_id", "") or ""
    message_id = getattr(message, "message_id", "") or ""
    root_id = getattr(message, "root_id", None)
    thread_id = str(root_id) if isinstance(root_id, str) and root_id else None

    create_time_raw = getattr(message, "create_time", None)
    created_at: float | None = None
    if create_time_raw:
        try:
            ts = int(create_time_raw)
            created_at = ts / 1000.0 if ts > 1e12 else float(ts)
        except (ValueError, TypeError):
            pass

    sender_id_obj = getattr(sender, "sender_id", None)
    sender_id = ""
    if sender_id_obj is not None:
        for key in ("open_id", "user_id", "union_id"):
            value = getattr(sender_id_obj, key, None)
            if isinstance(value, str) and value:
                sender_id = value
                break

    if not chat_id or not content_text:
        return None

    return IMIncomingMessage(
        platform="feishu",
        channel_id=str(chat_id),
        user_id=sender_id or "unknown",
        text=content_text,
        thread_id=thread_id,
        message_id=str(message_id) if message_id else None,
        created_at=created_at,
        raw={**_to_dict(data), "chat_type": getattr(message, "chat_type", None)},
    )


def _parse_ws_message(payload: dict[str, Any]) -> IMIncomingMessage | None:
    """从原始 dict 解析消息（兼容旧版 SDK 或 webhook 透传格式）。"""
    header = payload.get("header")
    event_type = header.get("event_type") if isinstance(header, dict) else ""
    if event_type != "im.message.receive_v1":
        return None

    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    message = event.get("message")
    sender = event.get("sender")
    if not isinstance(message, dict) or not isinstance(sender, dict):
        return None

    sender_type = sender.get("sender_type")
    if isinstance(sender_type, str) and sender_type.lower() in {"app", "bot"}:
        return None
    message_type = message.get("message_type")
    if message_type not in {"text", "post"}:
        return None

    content_text = extract_feishu_text(message_type, message.get("content"))

    mentions = message.get("mentions") or []
    content_text = _strip_bot_mentions_from_dicts(content_text, mentions)

    chat_id = str(message.get("chat_id", ""))
    message_id = str(message.get("message_id", ""))
    root_id = message.get("root_id")
    thread_id = str(root_id) if isinstance(root_id, str) and root_id else None

    create_time_raw = message.get("create_time")
    created_at: float | None = None
    if create_time_raw:
        try:
            ts = int(create_time_raw)
            created_at = ts / 1000.0 if ts > 1e12 else float(ts)
        except (ValueError, TypeError):
            pass

    if not chat_id or not content_text:
        return None

    sender_id = ""
    sender_node = sender.get("sender_id")
    if isinstance(sender_node, dict):
        for key in ("open_id", "user_id", "union_id"):
            value = sender_node.get(key)
            if isinstance(value, str) and value:
                sender_id = value
                break

    return IMIncomingMessage(
        platform="feishu",
        channel_id=chat_id,
        user_id=sender_id or "unknown",
        text=content_text,
        thread_id=thread_id,
        message_id=message_id or None,
        created_at=created_at,
        raw=payload,
    )


def _parse_card_action_object(data: Any) -> IMIncomingMessage | None:
    """Convert a typed P2 card-action callback into an approval command."""

    event = getattr(data, "event", None)
    action = getattr(event, "action", None)
    value = getattr(action, "value", None)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, dict) or value.get("xingclaw_action") != "tool_approval":
        return None

    decision = str(value.get("decision") or "").lower()
    tool_call_id = str(value.get("tool_call_id") or "").strip()
    if decision not in {"approve", "reject"} or not tool_call_id:
        return None

    context = getattr(event, "context", None)
    chat_id = str(getattr(context, "open_chat_id", None) or "")
    card_message_id = str(getattr(context, "open_message_id", None) or "card")
    operator = getattr(event, "operator", None)
    user_id = str(
        getattr(operator, "open_id", None)
        or getattr(operator, "user_id", None)
        or "unknown"
    )
    if not chat_id:
        return None

    return IMIncomingMessage(
        platform="feishu",
        channel_id=chat_id,
        user_id=user_id,
        text=f"/{decision} {tool_call_id}",
        thread_id=str(value.get("thread_id") or "") or None,
        message_id=f"card:{card_message_id}:{user_id}:{decision}:{tool_call_id}",
        raw=_to_dict(data),
    )


def _strip_bot_mentions_from_objects(text: str, mentions: list[Any]) -> str:
    """从文本中移除 @机器人 占位符（lark-oapi 强类型对象）。"""
    for m in mentions:
        mentioned_type = getattr(m, "mentioned_type", "") or ""
        if mentioned_type.lower() != "bot":
            continue
        key = getattr(m, "key", "") or ""
        if key and key in text:
            text = text.replace(key, "")
    return text.strip()


def _strip_bot_mentions_from_dicts(text: str, mentions: list[dict[str, Any]]) -> str:
    """从文本中移除 @机器人 占位符（dict 格式）。"""
    for m in mentions:
        if not isinstance(m, dict):
            continue
        mentioned_type = m.get("mentioned_type", "") or ""
        if mentioned_type.lower() != "bot":
            continue
        key = m.get("key", "") or ""
        if key and key in text:
            text = text.replace(key, "")
    return text.strip()


def _to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(payload, attr, None)
        if callable(fn):
            try:
                obj = fn()
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    for attr in ("raw_body", "raw", "body"):
        value = getattr(payload, attr, None)
        if isinstance(value, (bytes, str)):
            try:
                obj = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}


def _import_lark_sdk():
    try:
        import lark_oapi as lark  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Feishu long connection requires `lark-oapi`. Install with: pip install lark-oapi"
        ) from exc
    return lark
