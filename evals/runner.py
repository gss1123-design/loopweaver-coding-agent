from __future__ import annotations

"""Run small deterministic Agent evaluations without a real API key.

Examples:
    $env:PYTHONPATH = "$PWD/src"
    python -m evals
"""

import argparse
import asyncio
import copy
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_core import AgentContext, AgentEvent, AgentLoopConfig, AgentTool, AgentToolResult, run_agent_loop
from ai.event_stream import AssistantMessageEventStream
from ai.models import get_model
from ai.types import AssistantMessage, TextContent, ToolCall, UserMessage
from coding_agent.tracing import RunTraceRecorder
from coding_agent.serde import message_to_dict
from .artifacts import ArtifactStore, canonical_hash


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    responses: tuple[AssistantMessage, ...]
    expected_text: str
    expected_tools: tuple[str, ...] = ()


@dataclass
class EvalResult:
    case: str
    passed: bool
    reason: str
    duration_ms: int
    tool_names: list[str]
    trace: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "passed": self.passed,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "tool_names": self.tool_names,
            "trace": self.trace,
        }


def _text_message(text: str, *, input_tokens: int = 20, output_tokens: int = 5) -> AssistantMessage:
    from ai.types import Usage

    return AssistantMessage(
        content=[TextContent(text=text)],
        provider="offline-eval",
        model="deterministic",
        usage=Usage(input=input_tokens, output=output_tokens, total_tokens=input_tokens + output_tokens),
        stop_reason="stop",
    )


def _tool_message() -> AssistantMessage:
    from ai.types import Usage

    return AssistantMessage(
        content=[ToolCall(id="eval_tc_1", name="get_value", arguments={})],
        provider="offline-eval",
        model="deterministic",
        usage=Usage(input=30, output=8, total_tokens=38),
        stop_reason="toolUse",
    )


def build_cases() -> list[EvalCase]:
    return [
        EvalCase(
            name="direct_answer",
            prompt="say pong",
            responses=(_text_message("pong"),),
            expected_text="pong",
        ),
        EvalCase(
            name="tool_roundtrip",
            prompt="look up the value",
            responses=(_tool_message(), _text_message("the value is 42", input_tokens=40, output_tokens=8)),
            expected_text="42",
            expected_tools=("get_value",),
        ),
    ]


def _make_stream_fn(
    responses: tuple[AssistantMessage, ...],
) -> Callable[[Any, Any, Any], AssistantMessageEventStream]:
    index = 0

    def stream_fn(_model: Any, _context: Any, _options: Any) -> AssistantMessageEventStream:
        nonlocal index
        if index < len(responses):
            message = copy.deepcopy(responses[index])
            index += 1
        else:
            message = _text_message("unexpected extra model call")
            message.stop_reason = "error"
            message.error_message = "offline eval exhausted fake responses"

        stream = AssistantMessageEventStream()

        async def produce() -> None:
            stream.push({"type": "start", "partial": message})
            text = "".join(block.text for block in message.content if isinstance(block, TextContent))
            if text:
                stream.push(
                    {
                        "type": "text_delta",
                        "delta": text,
                        "partial": message,
                    }
                )
            stream.push({"type": "done", "partial": message})
            stream.end(message)

        asyncio.create_task(produce())
        return stream

    return stream_fn


def _get_eval_tools() -> list[AgentTool]:
    async def get_value(_tool_call_id: str, _params: dict[str, Any], _signal=None, _on_update=None) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="42")])

    return [
        AgentTool(
            name="get_value",
            label="Get value",
            description="Return the deterministic value 42 for offline evaluation.",
            parameters={"type": "object", "properties": {}},
            execute=get_value,
        )
    ]


async def run_case(case: EvalCase, artifacts: ArtifactStore | None = None, repetition: int = 1) -> EvalResult:
    events: list[AgentEvent] = []
    model = get_model("openai-standard", "gpt-4o-mini")
    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda messages: messages,
        max_turns=5,
    )
    context = AgentContext(
        system_prompt="You are an offline evaluation agent.",
        messages=[],
        tools=_get_eval_tools(),
    )

    started = time.perf_counter()
    try:
        messages = await run_agent_loop(
            prompts=[UserMessage(content=case.prompt)],
            context=context,
            config=config,
            emit=events.append,
            stream_fn=_make_stream_fn(case.responses),
        )
    except Exception as exc:
        if artifacts:
            artifacts.record({"layer": "harness_regression", "case_id": case.name,
                "input_hash": canonical_hash(case.prompt), "harness_id": "scripted-v1", "repetition": repetition,
                "outcome": "errored", "scores": {}, "telemetry": {}, "error": str(exc)},
                events=json.loads(json.dumps(events, default=lambda v: asdict(v) if is_dataclass(v) else str(v))))
        raise
    duration_ms = int((time.perf_counter() - started) * 1000)

    final = next((m for m in reversed(messages) if isinstance(m, AssistantMessage)), None)
    final_text = ""
    if final is not None:
        final_text = "".join(block.text for block in final.content if isinstance(block, TextContent))
    tool_names = [str(event.get("toolName")) for event in events if event.get("type") == "tool_execution_start"]

    recorder = RunTraceRecorder()
    trace: dict[str, Any] | None = None
    for event in events:
        trace = recorder.record_event(event)

    missing_tools = [name for name in case.expected_tools if name not in tool_names]
    if final is None:
        passed = False
        reason = "no final assistant message"
    elif final.stop_reason != "stop":
        passed = False
        reason = f"final stop_reason={final.stop_reason}"
    elif case.expected_text not in final_text:
        passed = False
        reason = f"expected text {case.expected_text!r} not found"
    elif missing_tools:
        passed = False
        reason = f"missing tools: {', '.join(missing_tools)}"
    else:
        passed = True
        reason = "ok"

    result = EvalResult(
        case=case.name,
        passed=passed,
        reason=reason,
        duration_ms=duration_ms,
        tool_names=tool_names,
        trace=trace,
    )
    if artifacts:
        artifacts.record({"layer": "harness_regression", "benchmark_version": 1,
            "case_id": case.name, "input_hash": canonical_hash(case.prompt), "harness_id": "scripted-v1",
            "repetition": repetition, "outcome": "scored", "scores": {"correctness": int(passed)},
            "telemetry": {"duration_ms": duration_ms, "tokens": trace.get("total_tokens") if trace else None,
                          "cost": None}, "reason": reason},
            events=json.loads(json.dumps(events, default=lambda v: asdict(v) if is_dataclass(v) else str(v))),
            messages=[message_to_dict(m) for m in messages])
    return result


async def run_suite(case_name: str | None = None, *, artifacts: ArtifactStore | None = None, repetitions: int = 1) -> list[EvalResult]:
    cases = [case for case in build_cases() if case_name is None or case.name == case_name]
    if case_name is not None and not cases:
        raise ValueError(f"Unknown evaluation case: {case_name}")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    return [await run_case(case, artifacts, repetition) for repetition in range(1, repetitions + 1) for case in cases]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run XingClaw deterministic offline evaluations")
    parser.add_argument("--case", default=None, help="Run one case by name")
    parser.add_argument("--artifacts", default=".eval", help="Directory for isolated run artifacts")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        store = ArtifactStore(args.artifacts, {"provider": "scripted", "benchmark_version": 1})
        results = asyncio.run(run_suite(args.case, artifacts=store, repetitions=args.repetitions))
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    payload = {
        "passed": all(result.passed for result in results),
        "total": len(results),
        "passed_count": sum(result.passed for result in results),
        "results": [result.to_dict() for result in results],
    }
    payload["artifact_dir"] = str(store.root)
    store.write("summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
