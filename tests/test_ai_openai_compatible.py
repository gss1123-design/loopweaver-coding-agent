from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.models import get_model
from ai.providers.openai_compatible import stream_openai_compatible
from ai.types import Context, UserMessage, StreamOptions


class _UnbufferedErrorResponse:
    status_code = 401
    reason_phrase = "Unauthorized"
    is_error = True
    is_closed = False

    def __init__(self) -> None:
        self.request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        self._content: bytes | None = None

    async def aread(self) -> bytes:
        self._content = b'{"error":{"message":"invalid api key"}}'
        return self._content

    @property
    def text(self) -> str:
        if self._content is None:
            raise httpx.ResponseNotRead()
        return self._content.decode("utf-8")

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError("401", request=self.request, response=self)  # type: ignore[arg-type]


class _StreamContext:
    def __init__(self, response: _UnbufferedErrorResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _UnbufferedErrorResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeAsyncClient:
    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def __init__(self, **kwargs) -> None:
        _ = kwargs
        self.response = _UnbufferedErrorResponse()

    def stream(self, *args, **kwargs) -> _StreamContext:
        _ = args, kwargs
        return _StreamContext(self.response)


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_deepseek_thinking_mode_is_explicit_and_does_not_change_defaults(self) -> None:
        payloads = []
        class Client(_FakeAsyncClient):
            def stream(self,*args,**kwargs):
                payloads.append(kwargs["json"])
                return super().stream(*args,**kwargs)
        async def exercise():
            base = get_model("deepseek","deepseek-chat")
            for model in (base,replace(base,compat={"thinking":"disabled"})):
                s=stream_openai_compatible(model,Context(messages=[UserMessage(content="hello")]),StreamOptions(api_key="test-key"))
                await s.result()
        with patch("ai.providers.openai_compatible.httpx.AsyncClient",Client):
            asyncio.run(exercise())
        self.assertNotIn("thinking",payloads[0])
        self.assertEqual(payloads[1]["thinking"],{"type":"disabled"})

    def test_unread_http_error_body_becomes_normal_stream_error(self) -> None:
        async def exercise():
            stream = stream_openai_compatible(
                get_model("deepseek", "deepseek-chat"),
                Context(messages=[UserMessage(content="hello")]),
            )
            events = [event async for event in stream]
            return events, await stream.result()

        with patch("ai.providers.openai_compatible.httpx.AsyncClient", _FakeAsyncClient):
            events, result = asyncio.run(exercise())

        self.assertEqual(result.stop_reason, "error")
        self.assertIn("HTTP 401", result.error_message or "")
        self.assertIn("invalid api key", result.error_message or "")
        self.assertTrue(any(event.get("type") == "error" for event in events))


if __name__ == "__main__":
    unittest.main()
