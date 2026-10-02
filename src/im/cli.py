from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .events import IMEventWatcher, IMEventWatcherOptions
from .feishu import FeishuAdapter, FeishuAdapterConfig
from .feishu_longconn import FeishuLongConnOptions, run_feishu_long_connection
from .server import IMServerOptions, run_http_server
from .service import IMService, IMServiceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoopWeaver IM bridge service")
    parser.add_argument("--platform", choices=["feishu"], default="feishu", help="IM platform (current: feishu)")
    parser.add_argument(
        "--transport",
        choices=["webhook", "longconn"],
        default="webhook",
        help="IM transport mode: webhook or long connection",
    )
    parser.add_argument("--workspace", default=".", help="Workspace path")
    parser.add_argument("--tool-backend", choices=["local", "docker"], default="local")
    parser.add_argument("--sandbox-image", default="loopweaver-sandbox:local")
    parser.add_argument("--structured-memory", action="store_true")
    parser.add_argument("--subagent-read-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", help="Webhook server host")
    parser.add_argument("--port", type=int, default=8787, help="Webhook server port")
    parser.add_argument("--path", default="/feishu/events", help="Webhook path")
    parser.add_argument("--provider", default="openai-standard", help="coding_agent provider")
    parser.add_argument("--model-id", default="gpt-4o-mini", help="coding_agent model id")
    parser.add_argument("--read-only", action="store_true", help="Enable read-only tool mode")
    parser.add_argument("--channel-queue-limit", type=int, default=20, help="Per-channel in-memory queue limit")
    parser.add_argument("--max-turns", type=int, default=50, help="Maximum model/tool turns per message")
    parser.add_argument("--max-context-messages", type=int, default=24, help="Compact IM history after this many messages")
    parser.add_argument("--max-context-tokens", type=int, default=12000, help="Approximate IM context token limit")
    parser.add_argument("--retain-recent-messages", type=int, default=8, help="Recent messages kept after compaction")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Maximum model output tokens per request")
    parser.add_argument(
        "--llm-compaction",
        action="store_true",
        help="Use an extra LLM request to summarize old IM history (higher quality, slower/costlier)",
    )
    parser.add_argument(
        "--tool-approval",
        action="store_true",
        help="Require interactive-card or /approve approval before executing risky tools",
    )
    parser.add_argument(
        "--approval-timeout-seconds",
        type=float,
        default=300.0,
        help="Seconds to wait for a tool approval before rejecting it",
    )
    parser.add_argument(
        "--enrich-prompt-context",
        action="store_true",
        help="Fetch Feishu user/chat names and add them to prompts (slower; off by default)",
    )
    parser.add_argument(
        "--stale-event-seconds",
        type=float,
        default=300.0,
        help="Drop events older than this age; use 0 to disable stale filtering",
    )
    parser.add_argument(
        "--events-dir",
        default="",
        help="Optional event directory for immediate/one-shot/periodic IM messages",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for IM bridge",
    )

    parser.add_argument("--feishu-app-id", default="", help="Feishu app_id")
    parser.add_argument("--feishu-app-secret", default="", help="Feishu app_secret")
    parser.add_argument("--feishu-verify-token", default="", help="Feishu event verify token (optional)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.platform == "feishu":
        if not args.feishu_app_id or not args.feishu_app_secret:
            parser.error("--feishu-app-id and --feishu-app-secret are required for feishu platform")
        adapter = FeishuAdapter(
            FeishuAdapterConfig(
                app_id=args.feishu_app_id,
                app_secret=args.feishu_app_secret,
                verify_token=args.feishu_verify_token or None,
            )
        )
    else:  # pragma: no cover
        parser.error(f"Unsupported platform: {args.platform}")
        return 2

    service = IMService(
        adapter=adapter,
        config=IMServiceConfig(
            workspace_dir=args.workspace,
            tool_backend=args.tool_backend,
            sandbox_image=args.sandbox_image,
            enable_structured_memory=args.structured_memory,
            subagent_read_only=args.subagent_read_only,
            provider=args.provider,
            model_id=args.model_id,
            read_only_mode=bool(args.read_only),
            channel_queue_limit=max(1, int(args.channel_queue_limit)),
            max_turns=max(1, int(args.max_turns)),
            max_context_messages=max(2, int(args.max_context_messages)) if args.max_context_messages else None,
            max_context_tokens=max(1, int(args.max_context_tokens)) if args.max_context_tokens else None,
            retain_recent_messages=max(2, int(args.retain_recent_messages)),
            max_output_tokens=max(1, int(args.max_output_tokens)) if args.max_output_tokens else None,
            llm_compaction=bool(args.llm_compaction),
            tool_approval=bool(args.tool_approval),
            approval_timeout_seconds=max(1.0, float(args.approval_timeout_seconds)),
            enrich_prompt_context=bool(args.enrich_prompt_context),
            stale_event_seconds=max(0.0, float(args.stale_event_seconds)),
        ),
    )
    events_dir = Path(args.events_dir) if args.events_dir else Path(args.workspace) / ".loopweaver" / "im" / "events"
    watcher = IMEventWatcher(service, IMEventWatcherOptions(events_dir=events_dir))
    watcher.start()
    try:
        if args.transport == "longconn":
            run_feishu_long_connection(
                service,
                FeishuLongConnOptions(
                    app_id=args.feishu_app_id,
                    app_secret=args.feishu_app_secret,
                    log_level=args.log_level,
                ),
            )
        else:
            server_options = IMServerOptions(host=args.host, port=args.port, path=args.path)
            run_http_server(service, server_options)
    except KeyboardInterrupt:
        print("\n[im] stopped")
    finally:
        watcher.stop()
        service.close()
    return 0
