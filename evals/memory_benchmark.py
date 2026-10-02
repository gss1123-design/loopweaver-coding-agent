"""Paired scripted Agent runs through real memory tools, not live LLM scores."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import tempfile
import time

from agent_core import AgentContext, AgentLoopConfig, AgentTool, AgentToolResult, run_agent_loop
from ai.event_stream import AssistantMessageEventStream
from ai.models import get_model
from ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from coding_agent.memory import MemoryStore, create_memory_tools
from coding_agent.serde import message_to_dict
from .artifacts import ArtifactStore, canonical_hash, compare_paired


@dataclass(frozen=True)
class Case:
    name: str
    key: str
    expected: str
    cached: str | None = None
    kind: str = "fact"
    scope: str = "workspace"
    expired: bool = False
    deleted: bool = False
    verify: bool = False


CASES = (
    Case("preference_hit","language","Python","Python","preference"),
    Case("decision_hit","database","SQLite","SQLite","decision"),
    Case("procedure_hit","test_command","python -m pytest","python -m pytest","procedure"),
    Case("cold_miss","formatter","ruff"),
    Case("expired_fact","port","9000","8000",expired=True),
    Case("deleted_fact","framework","FastAPI","Flask",deleted=True),
    Case("scope_isolation","language","Python","Java",scope="other-user"),
    Case("verify_stale_fact","version","v2","v1",verify=True),
)


async def run_case(case: Case, enabled: bool, store: ArtifactStore, repetition: int) -> dict:
    inputs = asdict(case)
    started = time.perf_counter()
    events, messages = [], []
    record = {"layer":"memory_benefit","benchmark_version":2,"case_id":case.name,
        "input_hash":canonical_hash(inputs),"inputs":inputs,"repetition":repetition,
        "harness_id":"memory-on" if enabled else "memory-off","outcome":"errored",
        "scores":{},"telemetry":{"tokens":None,"cost":None},
        "evaluation":"scripted_policy_real_agent_loop"}
    try:
        with tempfile.TemporaryDirectory() as workspace:
            memory = MemoryStore(workspace)
            if case.cached is not None:
                memory.put(case.scope,case.key,case.cached,kind=case.kind,source="fixture:prior-session")
                if case.expired:
                    with memory.connect() as db:
                        db.execute("UPDATE memories SET expires=?",(time.time()-1,))
                if case.deleted:
                    memory.delete(case.scope,case.key,source="fixture:user-delete")
            # Tool closure opens the persisted DB again, as a new session does.
            tools = [t for t in create_memory_tools(workspace,"workspace") if t.name == "memory_search"] if enabled else []
            async def find_source(call_id,params,signal=None,on_update=None):
                return AgentToolResult(content=[TextContent(text=json.dumps({"source":"project-config"}))])
            async def read_source(call_id,params,signal=None,on_update=None):
                return AgentToolResult(content=[TextContent(text=json.dumps({"value":case.expected}))])
            for name,execute in (("find_source",find_source),("read_source",read_source)):
                tools.append(AgentTool(name=name,label=name,description="Read a deterministic authoritative fixture",
                    parameters={"type":"object","properties":{}},execute=execute,read_only=True))
            request_count = 0
            def model_fn(model,context,options):
                nonlocal request_count
                request_count += 1
                results = [m for m in context.messages if isinstance(m,ToolResultMessage)]
                text = None
                if not results:
                    name = "memory_search" if enabled else "find_source"
                else:
                    last = results[-1]
                    data = json.loads("".join(b.text for b in last.content if isinstance(b,TextContent)))
                    if last.tool_name == "memory_search":
                        matches = [m for m in data if m["key"] == case.key]
                        if matches and not case.verify:
                            text = matches[0]["value"]
                        name = "find_source"
                    elif last.tool_name == "find_source":
                        name = "read_source"
                    else:
                        text = data["value"]
                        name = ""
                response = AssistantMessage(content=[TextContent(text=text)] if text is not None else
                    [ToolCall(id=f"call_{request_count}",name=name,arguments={"query":case.key} if name == "memory_search" else {})],
                    stop_reason="stop" if text is not None else "toolUse")
                stream = AssistantMessageEventStream()
                stream.push({"type":"start","partial":response})
                stream.push({"type":"done","partial":response})
                stream.end(response)
                return stream
            messages = await run_agent_loop(prompts=[UserMessage(content=f"Find {case.key}; verify={case.verify}")],
                context=AgentContext(system_prompt="Deterministic memory contract evaluation",messages=[],tools=tools),
                config=AgentLoopConfig(model=get_model("openai-standard","gpt-4o-mini"),convert_to_llm=lambda ms:ms,max_turns=6),
                emit=events.append,stream_fn=model_fn)
            final = next(m for m in reversed(messages) if isinstance(m,AssistantMessage))
            answer = "".join(b.text for b in final.content if isinstance(b,TextContent))
            tool_names = [e["toolName"] for e in events if e.get("type") == "tool_execution_start"]
            tool_messages = [m for m in messages if isinstance(m,ToolResultMessage)]
            record.update(outcome="scored",scores={"correctness":int(answer == case.expected and final.stop_reason == "stop")},
                evidence={"answer":answer,"expected":case.expected,"tool_names":tool_names,
                    "model_requests":request_count,"source_reads":tool_names.count("read_source"),
                    "tool_result_chars":sum(len(json.dumps(message_to_dict(m),ensure_ascii=False)) for m in tool_messages)})
            record["telemetry"].update(tool_calls=len(tool_names),model_requests=request_count)
    except Exception as exc:
        record["error"] = str(exc)
    record["telemetry"]["duration_ms"] = (time.perf_counter()-started)*1000
    store.record(record,events=json.loads(json.dumps(events,default=lambda x:asdict(x) if is_dataclass(x) else str(x))),
        messages=[message_to_dict(m) for m in messages])
    return record


async def run_memory_benchmark(store: ArtifactStore,repetitions: int = 2) -> dict:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    records = []
    for repetition in range(1,repetitions+1):
        for case in CASES:
            # Alternate arm order to avoid always timing the cold arm first.
            for enabled in ((False,True) if repetition % 2 else (True,False)):
                records.append(await run_case(case,enabled,store,repetition))
    comparisons = []
    for case in CASES:
        subset = [r for r in records if r["case_id"] == case.name]
        comparison = compare_paired(subset,"memory-off","memory-on")
        paired = {(r["repetition"],r["harness_id"]):r for r in subset if r["outcome"] == "scored"}
        deltas = [paired[(i,"memory-on")]["telemetry"]["tool_calls"]-paired[(i,"memory-off")]["telemetry"]["tool_calls"]
                  for i in range(1,repetitions+1) if (i,"memory-on") in paired and (i,"memory-off") in paired]
        comparisons.append({"case_id":case.name,**comparison,"tool_calls_mean_delta":sum(deltas)/len(deltas) if deltas else None})
    report = {"benchmark_version":2,"mode":"offline-scripted","records":records,"comparisons":comparisons,
        "passed":all(r["outcome"] == "scored" and r["scores"]["correctness"] == 1 for r in records),
        "limitations":"Policy chooses when to search/verify. No live LLM, no real token/cost measurement, no proof of semantic memory benefit or prompt-injection resistance. Duration includes fixture setup."}
    store.write("memory-report.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts",default="output/evals-memory")
    parser.add_argument("--repetitions",type=int,default=2)
    args = parser.parse_args()
    store = ArtifactStore(args.artifacts,{"benchmark_version":2,"mode":"offline-scripted-memory","repetitions":args.repetitions})
    report = asyncio.run(run_memory_benchmark(store,args.repetitions))
    print(json.dumps({"artifact_dir":str(store.root),"passed":report["passed"],"comparisons":report["comparisons"]},ensure_ascii=False,indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
