"""XingClaw 的本地 stdio MCP 示例服务器。

这个服务器只使用 Python 标准库，通过换行分隔的 JSON-RPC 与
``coding_agent.mcp.StdioMCPClient`` 通信。它提供三个只读工具：

* project_summary：查看目录摘要；
* python_outline：查看 Python 文件的类、函数和导入；
* git_log：查看最近的 Git 提交。

运行时 stdout 只能输出 JSON-RPC 响应，调试信息应写到 stderr。
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MAX_OUTPUT_CHARS = 12_000
DEFAULT_LIMIT = 50


def _root() -> Path:
    """把启动 MCP server 时的当前目录作为项目根目录。"""
    return Path.cwd().resolve()


def _resolve_workspace_path(path_text: str) -> Path:
    """解析相对项目路径，并拒绝通过 ``..`` 逃出项目根目录。"""
    root = _root()
    target = (root / (path_text or ".")).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path escapes MCP workspace boundary") from exc
    return target


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    """构造 MCP tools/call 的标准文本结果。"""
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n...<truncated>..."
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def _tool_definitions() -> list[dict[str, Any]]:
    """返回 MCP tools/list 的工具定义。"""
    return [
        {
            "name": "project_summary",
            "description": "Summarize files and directories under the project path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory, default ."},
                    "max_entries": {"type": "integer", "description": "Maximum entries, default 50"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "python_outline",
            "description": "List top-level imports, classes, and functions in a Python file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative Python file path"},
                    "max_items": {"type": "integer", "description": "Maximum outline items, default 100"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "git_log",
            "description": "Show recent Git commits for the project or one relative path.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional relative path"},
                    "limit": {"type": "integer", "description": "Maximum commits, default 10"},
                },
                "additionalProperties": False,
            },
        },
    ]


def _project_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    path_text = str(arguments.get("path", "."))
    target = _resolve_workspace_path(path_text)
    if not target.exists():
        return _text_result(f"Path not found: {path_text}", is_error=True)
    if not target.is_dir():
        return _text_result(f"Not a directory: {path_text}", is_error=True)

    limit = max(1, min(int(arguments.get("max_entries", DEFAULT_LIMIT)), 200))
    entries = []
    for item in sorted(target.iterdir(), key=lambda value: value.name.lower())[:limit]:
        suffix = "/" if item.is_dir() else ""
        size = "-" if item.is_dir() else str(item.stat().st_size)
        entries.append(f"{item.name}{suffix}\t{size}")
    return _text_result("\n".join(entries) if entries else "(empty)")


def _python_outline(arguments: dict[str, Any]) -> dict[str, Any]:
    path_text = str(arguments.get("path", "")).strip()
    if not path_text:
        return _text_result("Missing path", is_error=True)
    target = _resolve_workspace_path(path_text)
    if not target.exists():
        return _text_result(f"Path not found: {path_text}", is_error=True)
    if not target.is_file() or target.suffix.lower() != ".py":
        return _text_result(f"Not a Python file: {path_text}", is_error=True)

    try:
        tree = ast.parse(target.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return _text_result(f"Syntax error: {exc}", is_error=True)

    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = ", ".join(alias.name for alias in node.names)
            lines.append(f"{node.lineno}: import {names}")
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            names = ", ".join(alias.name for alias in node.names)
            lines.append(f"{node.lineno}: from {module} import {names}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            lines.append(f"{node.lineno}: class {node.name}({bases})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args = ast.unparse(node.args)
            lines.append(f"{node.lineno}: {prefix}def {node.name}({args})")

    limit = max(1, min(int(arguments.get("max_items", 100)), 300))
    return _text_result("\n".join(lines[:limit]) if lines else "(no top-level definitions)")


def _git_log(arguments: dict[str, Any]) -> dict[str, Any]:
    path_text = str(arguments.get("path", "")).strip()
    relative_path = ""
    if path_text:
        target = _resolve_workspace_path(path_text)
        if not target.exists():
            return _text_result(f"Path not found: {path_text}", is_error=True)
        relative_path = target.relative_to(_root()).as_posix()

    limit = max(1, min(int(arguments.get("limit", 10)), 100))
    command = ["git", "-C", str(_root()), "log", f"-{limit}", "--oneline", "--decorate"]
    if relative_path:
        command.extend(["--", relative_path])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return _text_result(f"git log failed: {exc}", is_error=True)

    output = completed.stdout.strip()
    if completed.stderr.strip():
        output += f"\n[stderr]\n{completed.stderr.strip()}"
    if completed.returncode != 0:
        return _text_result(output or f"git log exited with code {completed.returncode}", is_error=True)
    return _text_result(output or "(no commits)")


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "project_summary":
            return _project_summary(arguments)
        if name == "python_outline":
            return _python_outline(arguments)
        if name == "git_log":
            return _git_log(arguments)
        return _text_result(f"Unknown tool: {name}", is_error=True)
    except (TypeError, ValueError, OSError) as exc:
        return _text_result(str(exc), is_error=True)


def _handle(request: dict[str, Any]) -> dict[str, Any] | None:
    """处理一条 JSON-RPC 请求；通知没有 response。"""
    method = request.get("method")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "xingclaw-project", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_definitions()}}
    if method == "tools/call":
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        name = str(params.get("name", ""))
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        return {"jsonrpc": "2.0", "id": request_id, "result": _call_tool(name, arguments)}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            response = _handle(request if isinstance(request, dict) else {})
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
        except (json.JSONDecodeError, TypeError) as exc:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": str(exc)},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
