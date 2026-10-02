from __future__ import annotations

import json
import logging
import time
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from .feishu_content import extract_feishu_text
from .types import (
    IMChannelInfo,
    IMCardButton,
    IMIncomingMessage,
    IMOutgoingCard,
    IMOutgoingText,
    IMUserInfo,
    IMWebhookResult,
)

logger = logging.getLogger("loopweaver.im.feishu")


@dataclass
class FeishuAdapterConfig:
    app_id: str
    app_secret: str
    verify_token: str | None = None
    api_base: str = "https://open.feishu.cn/open-apis"
    request_timeout_seconds: int = 15
    user_cache_ttl: float = 3600.0
    chat_cache_ttl: float = 3600.0


class FeishuAdapter:
    """飞书适配器：事件接收 + 文本/卡片回复 + 消息更新 + 用户/频道缓存。"""

    def __init__(self, config: FeishuAdapterConfig) -> None:
        self.config = config
        self._token: str | None = None
        self._token_expire_at: float = 0
        self._token_lock = threading.Lock()
        # value 为 None 也要缓存：没有通讯录权限时，避免每条消息都重复请求并打印 400。
        self._user_cache: dict[str, tuple[IMUserInfo | None, float]] = {}
        self._chat_cache: dict[str, tuple[IMChannelInfo, float]] = {}
        self._http: Any | None = None
        self._http_lock = threading.Lock()

    def _get_http(self) -> Any:
        """复用同一个 httpx.Client，避免每次请求都重做 TLS 握手。

        原先每处调用都 `with httpx.Client(...)`，每个请求都要新建连接 + 完整
        TLS 握手，实测固定开销 2.7 秒。连接池复用后降到几百毫秒。
        """
        if self._http is not None:
            return self._http
        with self._http_lock:
            if self._http is None:
                httpx = _import_httpx()
                self._http = httpx.Client(
                    timeout=self.config.request_timeout_seconds,
                    limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
                )
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    # ------------------------------------------------------------------
    # Webhook 解析
    # ------------------------------------------------------------------

    def handle_webhook(self, headers: Mapping[str, str], body: bytes) -> IMWebhookResult:
        payload = self._parse_json(body)

        if payload.get("type") == "url_verification":
            challenge = str(payload.get("challenge", ""))
            logger.info("feishu url_verification success")
            return IMWebhookResult(ack={"challenge": challenge}, messages=[])

        if self.config.verify_token:
            token = payload.get("token")
            if token != self.config.verify_token:
                logger.warning("feishu verify_token mismatch")
                return IMWebhookResult(ack={"code": 19021, "msg": "invalid token"}, messages=[])

        # Interactive-card callbacks use a different payload shape from
        # im.message.receive_v1.  Convert a LoopWeaver approval button click to
        # the same internal command used by the text fallback.
        card_action = self._parse_card_action(payload)
        if card_action is not None:
            return IMWebhookResult(
                ack={"toast": {"type": "info", "content": "审批请求已提交"}},
                messages=[card_action],
            )

        header = payload.get("header")
        event_type = header.get("event_type") if isinstance(header, dict) else ""
        if event_type != "im.message.receive_v1":
            logger.info("ignored feishu event_type=%s", event_type)
            return IMWebhookResult(ack={"code": 0}, messages=[])

        event = payload.get("event")
        if not isinstance(event, dict):
            return IMWebhookResult(ack={"code": 0}, messages=[])

        message = event.get("message")
        sender = event.get("sender")
        if not isinstance(message, dict) or not isinstance(sender, dict):
            return IMWebhookResult(ack={"code": 0}, messages=[])

        sender_type = sender.get("sender_type")
        if isinstance(sender_type, str) and sender_type.lower() in {"app", "bot"}:
            return IMWebhookResult(ack={"code": 0}, messages=[])

        message_type = message.get("message_type")
        if message_type not in {"text", "post"}:
            logger.info("ignored non-text message_type=%s", message.get("message_type"))
            return IMWebhookResult(ack={"code": 0}, messages=[])

        content_text = extract_feishu_text(message_type, message.get("content"))

        mentions = message.get("mentions") or []
        for m in mentions:
            if not isinstance(m, dict):
                continue
            if (m.get("mentioned_type") or "").lower() != "bot":
                continue
            key = m.get("key", "") or ""
            if key and key in content_text:
                content_text = content_text.replace(key, "")
        content_text = content_text.strip()

        chat_id = str(message.get("chat_id", ""))
        message_id = str(message.get("message_id", ""))
        root_id = message.get("root_id")
        thread_id = str(root_id) if isinstance(root_id, str) and root_id else None

        sender_id = ""
        sender_node = sender.get("sender_id")
        if isinstance(sender_node, dict):
            for key in ("open_id", "user_id", "union_id"):
                value = sender_node.get(key)
                if isinstance(value, str) and value:
                    sender_id = value
                    break

        if not chat_id or not content_text:
            logger.warning("invalid feishu message: missing chat_id or content")
            return IMWebhookResult(ack={"code": 0}, messages=[])

        incoming = IMIncomingMessage(
            platform="feishu",
            channel_id=chat_id,
            user_id=sender_id or "unknown",
            text=content_text,
            thread_id=thread_id,
            message_id=message_id or None,
            raw=payload,
        )
        return IMWebhookResult(ack={"code": 0}, messages=[incoming])

    # ------------------------------------------------------------------
    # 发送文本消息（支持回复 + 线程）
    # ------------------------------------------------------------------

    def send_text(self, message: IMOutgoingText) -> str | None:
        httpx = _import_httpx()
        token = self._get_tenant_access_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
        content = json.dumps({"text": message.text}, ensure_ascii=False)

        if message.reply_to_message_id:
            return self._reply_message(httpx, headers, message.reply_to_message_id, "text", content)

        payload: dict[str, Any] = {
            "receive_id": message.channel_id,
            "msg_type": "text",
            "content": content,
        }
        url = f"{self.config.api_base}/im/v1/messages?receive_id_type=chat_id"
        return self._post_message(httpx, headers, url, payload)

    # ------------------------------------------------------------------
    # 更新已发送消息（流式打字效果）
    # ------------------------------------------------------------------

    def update_text(self, message_id: str, text: str) -> None:
        """更新已发送的卡片消息内容（飞书只支持 PATCH 卡片消息）。"""
        token = self._get_tenant_access_token()
        url = f"{self.config.api_base}/im/v1/messages/{message_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
        card = self._build_card_content("LoopWeaver", text)
        payload = {
            "content": card,
        }
        client = self._get_http()
        resp = client.patch(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            logger.warning(
                "feishu update_text %d message_id=%s body=%s",
                resp.status_code, message_id, err_body,
            )
            resp.raise_for_status()
        body = resp.json()
        if body.get("code") not in (0, "0", None):
            logger.warning("feishu update message failed message_id=%s: %s", message_id, body)

    # ------------------------------------------------------------------
    # 发送交互卡片消息（Markdown 富文本）
    # ------------------------------------------------------------------

    def send_card(self, message: IMOutgoingCard) -> str | None:
        httpx = _import_httpx()
        token = self._get_tenant_access_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}

        content = self._build_card_content(
            message.title,
            message.markdown_content,
            buttons=message.buttons,
        )

        if message.reply_to_message_id:
            return self._reply_message(httpx, headers, message.reply_to_message_id, "interactive", content)

        payload: dict[str, Any] = {
            "receive_id": message.channel_id,
            "msg_type": "interactive",
            "content": content,
        }
        url = f"{self.config.api_base}/im/v1/messages?receive_id_type=chat_id"
        return self._post_message(httpx, headers, url, payload)

    @staticmethod
    def _build_card_content(
        title: str,
        markdown: str,
        *,
        buttons: list[IMCardButton] | None = None,
    ) -> str:
        elements: list[dict[str, Any]] = [{"tag": "markdown", "content": markdown}]
        if buttons:
            elements.append(
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": button.text},
                            "type": button.style,
                            "value": button.value,
                        }
                        for button in buttons
                    ],
                }
            )
        card = {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": title}},
            "elements": elements,
        }
        return json.dumps(card, ensure_ascii=False)

    @staticmethod
    def _parse_card_action(payload: dict[str, Any]) -> IMIncomingMessage | None:
        """Parse the HTTP card-callback shape used by Feishu/Lark."""

        action = payload.get("action")
        if not isinstance(action, dict):
            event = payload.get("event")
            action = event.get("action") if isinstance(event, dict) else None
            source = event if isinstance(event, dict) else payload
        else:
            source = payload
        if not isinstance(action, dict):
            return None
        value = action.get("value")
        if not isinstance(value, dict) or value.get("loopweaver_action") != "tool_approval":
            return None

        decision = str(value.get("decision") or "").lower()
        tool_call_id = str(value.get("tool_call_id") or "").strip()
        if decision not in {"approve", "reject"} or not tool_call_id:
            return None

        operator = source.get("operator")
        operator = operator if isinstance(operator, dict) else {}
        user_id = str(
            operator.get("open_id")
            or source.get("open_id")
            or source.get("user_id")
            or "unknown"
        )
        context = source.get("context")
        context = context if isinstance(context, dict) else {}
        chat_id = str(context.get("open_chat_id") or source.get("open_chat_id") or "")
        card_message_id = str(
            context.get("open_message_id") or source.get("open_message_id") or "card"
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
            created_at=time.time(),
            raw=payload,
        )

    # ------------------------------------------------------------------
    # 用户 / 频道信息缓存
    # ------------------------------------------------------------------

    def get_user_info(self, user_id: str) -> IMUserInfo | None:
        now = time.time()
        cached = self._user_cache.get(user_id)
        if cached and now - cached[1] < self.config.user_cache_ttl:
            return cached[0]

        token = self._get_tenant_access_token()
        url = f"{self.config.api_base}/contact/v3/users/{user_id}?user_id_type=open_id"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = self._get_http().get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") not in (0, "0"):
                logger.warning("feishu get_user_info failed user_id=%s: %s", user_id, data)
                self._user_cache[user_id] = (None, now)
                return None
            user = data.get("data", {}).get("user", {})
            info = IMUserInfo(
                user_id=user_id,
                name=user.get("name", ""),
                avatar_url=user.get("avatar", {}).get("avatar_72", ""),
                department=", ".join(user.get("department_ids", [])),
            )
            self._user_cache[user_id] = (info, now)
            return info
        except Exception as exc:
            logger.warning("feishu get_user_info error user_id=%s: %s", user_id, exc)
            self._user_cache[user_id] = (None, now)
            return None

    def get_chat_info(self, chat_id: str) -> IMChannelInfo | None:
        now = time.time()
        cached = self._chat_cache.get(chat_id)
        if cached and now - cached[1] < self.config.chat_cache_ttl:
            return cached[0]

        token = self._get_tenant_access_token()
        url = f"{self.config.api_base}/im/v1/chats/{chat_id}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = self._get_http().get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") not in (0, "0"):
                logger.warning("feishu get_chat_info failed chat_id=%s: %s", chat_id, data)
                return None
            chat = data.get("data", {})
            info = IMChannelInfo(
                channel_id=chat_id,
                name=chat.get("name", ""),
                description=chat.get("description", ""),
                owner_id=chat.get("owner_id", ""),
                member_count=int(chat.get("user_count", 0)),
            )
            self._chat_cache[chat_id] = (info, now)
            return info
        except Exception as exc:
            logger.warning("feishu get_chat_info error chat_id=%s: %s", chat_id, exc)
            return None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reply_message(self, httpx, headers: dict, message_id: str, msg_type: str, content: str) -> str | None:
        """使用飞书 Reply API 回复指定消息。"""
        url = f"{self.config.api_base}/im/v1/messages/{message_id}/reply"
        payload = {"msg_type": msg_type, "content": content}
        resp = self._get_http().post(url, headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") not in (0, "0", None):
            raise RuntimeError(f"Feishu reply message failed: {body}")
        return body.get("data", {}).get("message_id")

    def _post_message(self, httpx, headers: dict, url: str, payload: dict) -> str | None:
        """发送新消息并返回 message_id。"""
        resp = self._get_http().post(url, headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") not in (0, "0", None):
            raise RuntimeError(f"Feishu send message failed: {body}")
        msg_id = body.get("data", {}).get("message_id")
        logger.info("feishu message sent message_id=%s", msg_id)
        return msg_id

    def _get_tenant_access_token(self) -> str:
        with self._token_lock:
            now = time.time()
            if self._token and now < self._token_expire_at - 60:
                return self._token

            url = f"{self.config.api_base}/auth/v3/tenant_access_token/internal"
            payload = {"app_id": self.config.app_id, "app_secret": self.config.app_secret}
            response = self._get_http().post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get("code") not in (0, "0"):
                raise RuntimeError(f"Feishu auth failed: {data}")
            token = data.get("tenant_access_token")
            expire = data.get("expire")
            if not isinstance(token, str) or not token:
                raise RuntimeError("Feishu auth response missing tenant_access_token")
            ttl = int(expire) if isinstance(expire, int) and expire > 0 else 7200
            self._token = token
            self._token_expire_at = now + ttl
            logger.info("feishu tenant_access_token refreshed ttl_seconds=%d", ttl)
            return token

    @staticmethod
    def _parse_json(body: bytes) -> dict[str, Any]:
        try:
            raw = json.loads(body.decode("utf-8"))
        except Exception:
            logger.exception("failed to parse webhook json body")
            return {}
        if isinstance(raw, dict):
            return raw
        return {}


def _import_httpx():
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError("httpx is required for feishu adapter") from exc
    return httpx
