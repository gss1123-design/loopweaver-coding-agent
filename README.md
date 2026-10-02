# LoopWeaver Coding Agent

基于 Python 的 AI 编程助手，包含统一 LLM 接口、Agent 执行循环、编程工具、会话管理和飞书集成。

## 功能

- 统一接入 Anthropic、OpenAI 兼容接口和 DeepSeek，支持流式响应。
- 通过 Agent 循环调用文件读取、写入、编辑、命令执行与代码检索工具。
- 支持会话持久化、恢复、分叉、上下文压缩与失败重试。
- 提供工具审批、执行追踪、扩展、Skills 和 stdio MCP 集成。
- 支持子 Agent 工作区隔离、变更合入与任务恢复，以及可选 Docker 工具后端。
- 提供可选结构化记忆与记忆检索评测。
- 通过飞书 Webhook 或长连接接收消息、执行任务并回复结果。

## 快速开始

需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/gss1123-design/loopweaver-coding-agent.git
cd loopweaver-coding-agent
python -m pip install -e ".[dev]"
```

设置所选模型提供方的环境变量后运行：

```bash
export ANTHROPIC_API_KEY="your_api_key"
python -m coding_agent --mode interactive --provider anthropic --model-id claude-sonnet-4-5
```

Windows PowerShell：

```powershell
Copy-Item .env.ps1.example .env.ps1
# 编辑 .env.ps1，填写实际凭据。
. .\.env.ps1
python -m coding_agent --mode interactive --provider anthropic --model-id claude-sonnet-4-5
```

也可以使用 `./dev.sh --mode cli` 或 `.\dev.ps1 -Mode cli`；开发脚本会加载对应的本地环境文件。

单次任务：

```bash
python -m coding_agent --mode print --provider deepseek --model-id deepseek-chat --prompt "概括当前项目的目录结构"
```

可用参数见 `python -m coding_agent --help`。已有的 Python 包名和命令 `xingclaw`、`xingclaw-im` 保留兼容。

## 飞书集成

安装可选依赖，并配置 `FEISHU_APP_ID`、`FEISHU_APP_SECRET` 和所选 LLM 的 API Key：

```bash
python -m pip install -e ".[dev,feishu]"
```

```powershell
.\dev.ps1 -Mode im -Transport longconn -ToolApproval
```

工具审批卡片需要在飞书开放平台启用 `card.action.trigger` 回调；具体启动选项见 `python -m im --help`。

## 测试与评测

```bash
python -m pytest -q
python -m evals --help
```

Docker 集成测试通过 `XINGCLAW_TEST_DOCKER=1` 显式开启。Docker 工具后端使用的镜像可以这样构建：

```bash
docker build -t xingclaw-sandbox:local tools/sandbox
```

## 项目结构

```text
src/
  ai/             统一模型接口与 Provider 实现
  agent_core/     Agent 生命周期与执行循环
  coding_agent/   编程工具、会话、扩展、记忆与隔离执行
  im/             飞书适配、消息路由与持久化
tests/            项目自动化测试
examples/         调用与 MCP 示例
evals/            Agent 与记忆评测代码
tools/sandbox/    可选 Docker 运行环境
```

本地运行配置可放在 `.xingclaw/`。MCP 示例见 [examples/mcp](examples/mcp/README.md)。
