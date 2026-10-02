from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from ai.types import TextContent
from agent_core import AgentTool, AgentToolResult


class MCPClient(Protocol):
    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """
        调用 MCP 服务器工具并返回结果对象。
        """


class MCPProtocolError(RuntimeError):
    """MCP server returned a JSON-RPC error or an invalid response."""


@dataclass
class MCPStdioServerConfig:
    """How to start one local MCP server over newline-delimited JSON-RPC."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None = None
    protocol_version: str = "2025-06-18"
    request_timeout: float = 30.0

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "MCPStdioServerConfig | None":
        name = raw.get("name")
        command = raw.get("command")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(command, str) or not command.strip():
            return None
        args = raw.get("args", [])
        env = raw.get("env", {})
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            args = []
        if not isinstance(env, dict):
            env = {}
        clean_env = {str(key): str(value) for key, value in env.items()}
        cwd = raw.get("cwd") if isinstance(raw.get("cwd"), str) else None
        protocol_version = raw.get("protocol_version", "2025-06-18")
        if not isinstance(protocol_version, str) or not protocol_version.strip():
            protocol_version = "2025-06-18"
        timeout = raw.get("request_timeout", 30.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = 30.0
        return cls(
            name=name.strip(),
            command=command.strip(),
            args=list(args),
            env=clean_env,
            cwd=cwd,
            protocol_version=protocol_version,
            request_timeout=float(timeout),
        )


@dataclass
class _StdioConnection:
    config: MCPStdioServerConfig
    process: asyncio.subprocess.Process
    request_lock: asyncio.Lock
    next_id: int = 1


class StdioMCPClient:
    """Small dependency-free MCP client for local stdio servers.

    The client implements the part XingClaw needs today:

    ``initialize`` -> ``notifications/initialized`` -> ``tools/list`` /
    ``tools/call``.

    Each configured server gets its own subprocess and JSON-RPC connection.
    Requests on one connection are serialized so response IDs cannot be mixed
    up; different servers can still be used concurrently.
    """

    def __init__(self, servers: list[dict[str, Any] | MCPStdioServerConfig]) -> None:
        configs: dict[str, MCPStdioServerConfig] = {}
        for raw in servers:
            config = raw if isinstance(raw, MCPStdioServerConfig) else MCPStdioServerConfig.from_raw(raw)
            if config is not None:
                configs[config.name] = config
        self._configs = configs
        self._connections: dict[str, _StdioConnection] = {}
        self._connections_lock = asyncio.Lock()
        self._ref_count = 0

    @property
    def server_names(self) -> list[str]:
        return sorted(self._configs)

    def retain(self) -> "StdioMCPClient":
        """Share this client with another session."""
        self._ref_count += 1
        return self

    def release(self) -> None:
        """Release one session reference and stop at the last owner."""
        self._ref_count = max(0, self._ref_count - 1)
        if self._ref_count == 0:
            self.close()

    async def _get_connection(self, server: str) -> _StdioConnection:
        existing = self._connections.get(server)
        if existing is not None and existing.process.returncode is None:
            return existing

        config = self._configs.get(server)
        if config is None:
            raise ValueError(f"Unknown MCP stdio server: {server}")

        async with self._connections_lock:
            existing = self._connections.get(server)
            if existing is not None and existing.process.returncode is None:
                return existing
            connection = await self._start_connection(config)
            self._connections[server] = connection
            return connection

    async def _start_connection(self, config: MCPStdioServerConfig) -> _StdioConnection:
        env = os.environ.copy()
        env.update(config.env)
        process = await asyncio.create_subprocess_exec(
            config.command,
            *config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=config.cwd,
            env=env,
        )
        connection = _StdioConnection(
            config=config,
            process=process,
            request_lock=asyncio.Lock(),
        )
        try:
            await self._request(
                connection,
                "initialize",
                {
                    "protocolVersion": config.protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "xingclaw", "version": "0.2.0"},
                },
            )
            await self._notify(connection, "notifications/initialized", {})
            return connection
        except Exception:
            await self._terminate_process(process)
            raise

    async def _notify(self, connection: _StdioConnection, method: str, params: dict[str, Any]) -> None:
        process = connection.process
        if process.stdin is None:
            raise MCPProtocolError("MCP server stdin is unavailable")
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def _request(
        self,
        connection: _StdioConnection,
        method: str,
        params: dict[str, Any],
    ) -> Any:
        async with connection.request_lock:
            process = connection.process
            if process.stdin is None or process.stdout is None:
                raise MCPProtocolError("MCP server pipes are unavailable")
            request_id = connection.next_id
            connection.next_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await process.stdin.drain()

            while True:
                raw_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=connection.config.request_timeout,
                )
                if not raw_line:
                    stderr_hint = ""
                    if process.returncode is not None:
                        stderr_hint = f" (process exited with code {process.returncode})"
                    raise MCPProtocolError(f"MCP server closed stdout{stderr_hint}")
                try:
                    response = json.loads(raw_line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise MCPProtocolError(f"Invalid MCP JSON-RPC response: {exc}") from exc
                if not isinstance(response, dict):
                    continue
                # Notifications have no id. They are valid while a request is
                # in flight, so ignore them and continue waiting for our id.
                if response.get("id") != request_id:
                    continue
                if "error" in response:
                    error = response.get("error")
                    if isinstance(error, dict):
                        message = str(error.get("message", "MCP request failed"))
                        code = error.get("code")
                        raise MCPProtocolError(f"MCP {method} failed ({code}): {message}")
                    raise MCPProtocolError(f"MCP {method} failed: {error}")
                return response.get("result")

    async def list_tools(self, server: str) -> list[dict[str, Any]]:
        """Discover tool schemas from one server, following pagination."""
        connection = await self._get_connection(server)
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = await self._request(connection, "tools/list", params)
            if not isinstance(result, dict):
                break
            page = result.get("tools")
            if isinstance(page, list):
                tools.extend(item for item in page if isinstance(item, dict))
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor
        return tools

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        connection = await self._get_connection(server)
        return await self._request(
            connection,
            "tools/call",
            {"name": tool, "arguments": arguments},
        )

    async def aclose(self) -> None:
        connections = list(self._connections.values())
        self._connections.clear()
        for connection in connections:
            await self._terminate_process(connection.process)

    def close(self) -> None:
        """Best-effort synchronous cleanup for AgentSession.close()."""
        for connection in list(self._connections.values()):
            process = connection.process
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                process.terminate()
        self._connections.clear()

    @staticmethod
    async def _terminate_process(process: asyncio.subprocess.Process) -> None:
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            if process.returncode is None:
                process.kill()


def create_stdio_mcp_client(
    servers: list[dict[str, Any]] | None,
) -> StdioMCPClient | None:
    """Create a stdio client only when at least one server has ``command``."""
    client = StdioMCPClient(servers or [])
    return client if client.server_names else None


@dataclass
class MCPToolConfig:
    name: str
    description: str
    parameters: dict[str, Any]
    server: str
    tool: str
    # MCP tools are treated conservatively unless the configuration explicitly
    # declares them read-only and safe to invoke without human approval.
    read_only: bool = False
    requires_approval: bool = True


def parse_mcp_tool_configs(raw_servers: list[dict[str, Any]] | None) -> list[MCPToolConfig]:
    if not raw_servers:
        return []
    result: list[MCPToolConfig] = []
    for server in raw_servers:
        if not isinstance(server, dict):
            continue
        server_name = server.get("name")
        tools = server.get("tools")
        if not isinstance(server_name, str) or not isinstance(tools, list):
            continue
        for item in tools:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            tool = item.get("tool") or name
            description = item.get("description") or f"MCP tool proxy: {server_name}.{tool}"
            params = item.get("parameters")
            if not isinstance(name, str) or not isinstance(tool, str):
                continue
            if not isinstance(description, str):
                description = str(description)
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}, "required": [], "additionalProperties": True}
            read_only = item.get("read_only") is True
            requires_approval = item.get("requires_approval")
            if not isinstance(requires_approval, bool):
                requires_approval = True
            result.append(
                MCPToolConfig(
                    name=name,
                    description=description,
                    parameters=params,
                    server=server_name,
                    tool=tool,
                    read_only=read_only,
                    requires_approval=requires_approval,
                )
            )
    return result


def create_mcp_proxy_tools(configs: list[MCPToolConfig], client: MCPClient | None) -> list[AgentTool]:
    tools: list[AgentTool] = []
    for cfg in configs:
        async def _execute(tool_call_id, params, signal=None, on_update=None, *, _cfg=cfg):  # type: ignore[no-untyped-def]
            _ = tool_call_id, signal, on_update
            args = params if isinstance(params, dict) else {}
            if client is None:
                return AgentToolResult(
                    content=[TextContent(text=f"MCP bridge unavailable for `{_cfg.name}`")],
                    is_error=True,
                )
            try:
                result = await client.call_tool(_cfg.server, _cfg.tool, args)
            except Exception as exc:  # pragma: no cover - adapter-specific
                return AgentToolResult(
                    content=[TextContent(text=f"MCP call failed `{_cfg.server}.{_cfg.tool}`: {exc}")],
                    is_error=True,
                )
            return AgentToolResult(
                content=[TextContent(text=_normalize_mcp_result(result))],
                details={"server": _cfg.server, "tool": _cfg.tool},
            )

        tools.append(
            AgentTool(
                name=cfg.name,
                label=f"MCP/{cfg.server}",
                description=cfg.description,
                parameters=cfg.parameters,
                execute=_execute,
                read_only=cfg.read_only,
                requires_approval=cfg.requires_approval,
            )
        )
    return tools


def _normalize_mcp_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item is not None:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts)
        structured = value.get("structuredContent")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False, default=str)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)
