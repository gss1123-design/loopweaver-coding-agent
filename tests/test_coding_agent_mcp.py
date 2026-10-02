from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coding_agent.factory import create_agent_session
from coding_agent.mcp import StdioMCPClient, create_mcp_proxy_tools, parse_mcp_tool_configs
from coding_agent.types import CreateAgentSessionOptions


class _FakeMCPClient:
    async def call_tool(self, server: str, tool: str, arguments: dict):
        return f"{server}.{tool}:{arguments.get('q', '')}"


class CodingAgentMCPTests(unittest.TestCase):
    def test_parse_mcp_configs_and_proxy_execute(self) -> None:
        cfg = parse_mcp_tool_configs(
            [
                {
                    "name": "demo",
                    "tools": [
                        {
                            "name": "mcp_search",
                            "tool": "search",
                            "description": "search via mcp",
                            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                        }
                    ],
                }
            ]
        )
        tools = create_mcp_proxy_tools(cfg, client=_FakeMCPClient())
        self.assertEqual(len(tools), 1)
        result = asyncio.run(tools[0].execute("tc1", {"q": "abc"}))
        self.assertIn("demo.search:abc", result.content[0].text if result.content else "")

    def test_mcp_tool_security_metadata_is_configurable(self) -> None:
        cfg = parse_mcp_tool_configs(
            [
                {
                    "name": "demo",
                    "tools": [
                        {
                            "name": "mcp_read",
                            "tool": "read",
                            "description": "read-only MCP tool",
                            "parameters": {"type": "object", "properties": {}},
                            "read_only": True,
                            "requires_approval": False,
                        }
                    ],
                }
            ]
        )

        self.assertTrue(cfg[0].read_only)
        self.assertFalse(cfg[0].requires_approval)
        proxy = create_mcp_proxy_tools(cfg, client=_FakeMCPClient())[0]
        self.assertTrue(proxy.read_only)
        self.assertFalse(proxy.requires_approval)

    def test_factory_registers_mcp_proxy_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    mcp_client=_FakeMCPClient(),
                    mcp_servers=[
                        {
                            "name": "demo",
                            "tools": [
                                {
                                    "name": "mcp_echo",
                                    "tool": "echo",
                                    "description": "echo from mcp",
                                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                                }
                            ],
                        }
                    ],
                )
            )
            names = {t.name for t in session.agent.state.tools}
            self.assertIn("mcp_echo", names)
            session.close()

    def test_stdio_client_initializes_lists_tools_and_calls_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server_path = Path(tmp_dir) / "fake_mcp_server.py"
            server_path.write_text(
                "\n".join(
                    [
                        "import json, sys",
                        "for line in sys.stdin:",
                        "    req = json.loads(line)",
                        "    method = req.get('method')",
                        "    if method == 'initialize':",
                        "        result = {'protocolVersion': req['params']['protocolVersion'], 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'fake', 'version': '1'}}",
                        "    elif method == 'notifications/initialized':",
                        "        continue",
                        "    elif method == 'tools/list':",
                        "        result = {'tools': [{'name': 'echo', 'description': 'echo text', 'inputSchema': {'type': 'object'}}]} ",
                        "    elif method == 'tools/call':",
                        "        value = req.get('params', {}).get('arguments', {}).get('text', '')",
                        "        result = {'content': [{'type': 'text', 'text': 'echo:' + str(value)}]} ",
                        "    else:",
                        "        result = {'content': []}",
                        "    if 'id' in req:",
                        "        print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': result}), flush=True)",
                    ]
                ),
                encoding="utf-8",
            )

            async def exercise() -> tuple[list[dict], object]:
                client = StdioMCPClient(
                    [
                        {
                            "name": "demo",
                            "command": sys.executable,
                            "args": [str(server_path)],
                        }
                    ]
                )
                try:
                    tools = await client.list_tools("demo")
                    result = await client.call_tool("demo", "echo", {"text": "hello"})
                    return tools, result
                finally:
                    await client.aclose()

            tools, result = asyncio.run(exercise())
            self.assertEqual(tools[0]["name"], "echo")
            self.assertEqual(result["content"][0]["text"], "echo:hello")

    def test_proxy_normalizes_mcp_content_blocks(self) -> None:
        class Client:
            async def call_tool(self, server: str, tool: str, arguments: dict):
                return {"content": [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]}

        cfg = parse_mcp_tool_configs(
            [{"name": "demo", "tools": [{"name": "echo", "tool": "echo", "parameters": {}}]}]
        )
        result = asyncio.run(create_mcp_proxy_tools(cfg, Client())[0].execute("tc", {}))
        self.assertEqual(result.content[0].text, "line one\nline two")

    def test_factory_auto_creates_stdio_client_from_server_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = create_agent_session(
                CreateAgentSessionOptions(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    mcp_servers=[
                        {
                            "name": "demo",
                            "command": sys.executable,
                            "args": ["fake_server.py"],
                            "tools": [
                                {
                                    "name": "mcp_echo",
                                    "tool": "echo",
                                    "parameters": {"type": "object"},
                                }
                            ],
                        }
                    ],
                )
            )
            self.assertIsInstance(session.mcp_client, StdioMCPClient)
            self.assertTrue(session.mcp_client_owned)
            self.assertIn("mcp_echo", {tool.name for tool in session.agent.state.tools})
            session.close()


if __name__ == "__main__":
    unittest.main()
