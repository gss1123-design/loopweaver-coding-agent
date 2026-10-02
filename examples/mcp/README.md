# XingClaw 本地 MCP 示例

这个目录提供一个只依赖 Python 标准库的 stdio MCP server，方便学习和验证
XingClaw 的 MCP 链路。

server 提供三个只读工具：

- `project_summary`：列出项目目录的条目和文件大小；
- `python_outline`：用 `ast` 查看 Python 文件的顶层导入、类和函数；
- `git_log`：查看最近 Git 提交。

## 启用配置

在项目根目录创建 `.xingclaw/settings.json`，写入以下配置：

```json
{
  "mcp_servers": [
    {
      "name": "local-project",
      "command": "python",
      "args": ["examples/mcp/xingclaw_project_server.py"],
      "tools": [
        {
          "name": "project_summary",
          "tool": "project_summary",
          "description": "Summarize the project directory",
          "parameters": {"type": "object", "properties": {}},
          "read_only": true,
          "requires_approval": false
        }
      ]
    }
  ]
}
```

从项目根目录启动，确保相对路径和 MCP server 的工作目录一致：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m coding_agent --mode interactive --workspace .
```

MCP server 是按需启动的：创建会话时只创建代理工具，模型第一次调用其中一个
工具时，XingClaw 才启动这个 Python 子进程并完成 `initialize`、
`notifications/initialized`、`tools/call`。

也可以单独验证 server 本身（输入一行 JSON-RPC）：

```powershell
'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' |
  python examples/mcp/xingclaw_project_server.py
```

不要让普通日志写到 stdout；MCP 客户端把 stdout 当作 JSON-RPC 通道，调试信息
应写到 stderr。
