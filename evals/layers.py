"""Fixed offline module-contract benchmarks, not a model capability score."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time

from .runner import build_cases, run_case
from .artifacts import ArtifactStore, canonical_hash, compare_paired
from ai.types import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from coding_agent.agent_session import AgentSession
from coding_agent.memory import MemoryStore
from coding_agent.recovery import batch_key, recover_completed_batch
from coding_agent.serde import message_to_dict


def record(store, layer, case, inputs, harness, repetition, check):
    started = time.perf_counter()
    try:
        passed, evidence = check()
        outcome, error = "scored", None
    except Exception as exc:
        passed, evidence, outcome, error = False, {}, "errored", str(exc)
    result = {"layer":layer,"case_id":case,"input_hash":canonical_hash(inputs),
        "benchmark_version":1,"harness_id":harness,"repetition":repetition,
        "outcome":outcome,"scores":{"correctness":int(passed)} if outcome == "scored" else {},
        "telemetry":{"duration_ms":(time.perf_counter()-started)*1000,"tokens":None,"cost":None},
        "inputs":inputs,"evidence":evidence,"error":error}
    store.record(result)
    return result


async def run_layers(store, repetitions=1):
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    contracts = []
    harness_passed = True
    for repetition in range(1,repetitions+1):
        for case in build_cases():
            harness_passed = (await run_case(case,store,repetition)).passed and harness_passed
        assistant = AssistantMessage(content=[ToolCall(id="a", name="read")],stop_reason="toolUse", timestamp=1)
        result = ToolResultMessage(tool_call_id="a",tool_name="read",timestamp=2)
        def context_contract():
            messages = [UserMessage(content="older"),assistant,result,UserMessage(content="latest")]
            older,recent = AgentSession._split_for_compaction(messages,retain=2)
            valid = not AgentSession._validate_message_sequence(older) and not AgentSession._validate_message_sequence(recent)
            return valid, {"older_count":len(older),"recent_count":len(recent)}
        contracts.append(record(store,"context_governance","tool_pair_boundary",{"retain":2},"runtime-v1",repetition,context_contract))
        journal = [{"kind":"tool_batch_committed","payload":{"batch_key":batch_key(assistant)}},
                   {"kind":"tool_result_committed","payload":{"batch_key":batch_key(assistant),"message":message_to_dict(result)}}]
        contracts.append(record(store,"recovery_correctness","committed_result_reuse",{"call":"read","id":"a"},"runtime-v1",repetition,
            lambda:(recover_completed_batch(assistant,[],journal)[0].tool_call_id == "a", {"tool_reexecutions":0})))
        def uncertain_contract():
            try:
                recover_completed_batch(assistant,[],[])
            except ValueError:
                return True,{"blocked":True}
            return False,{"blocked":False}
        contracts.append(record(store,"recovery_correctness","uncertain_result_blocks",{"id":"a"},"runtime-v1",repetition,uncertain_contract))
        def projection_contract():
            from ai.models import get_model
            from ai.types import TextContent
            from coding_agent.factory import create_agent_session
            from coding_agent.types import CreateAgentSessionOptions
            with tempfile.TemporaryDirectory() as workspace:
                options = CreateAgentSessionOptions(workspace_dir=workspace,session_id="benchmark",
                    model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False)
                session = create_agent_session(options)
                final = AssistantMessage(content=[TextContent(text="already completed")])
                # Event persistence is async; this benchmark runs inside an
                # event loop, so commit the same durable projection contract.
                session.store.append_journal_entry("message_checkpoint",{
                    "anchor_leaf":session.store.get_leaf_id(),"message":message_to_dict(final)})
                session.close()
                restored = create_agent_session(options)
                try:
                    status = restored.recovery_status()
                    return status["action"] == "project_final" and status["recoverable"],status
                finally:
                    restored.close()
        contracts.append(record(store,"recovery_correctness","restart_final_projection",{"boundary":"commit_before_projection"},"runtime-v2",repetition,projection_contract))
        with tempfile.TemporaryDirectory() as workspace:
            memory = MemoryStore(workspace)
            memory.put("user:a","language","Python",kind="preference",source="benchmark:fixture")
            for harness,enabled in (("memory-off",False),("memory-on",True)):
                def recall(enabled=enabled):
                    values = memory.search("user:a","language") if enabled else []
                    return any(v["value"] == "Python" for v in values), {"retrieved":values,"evaluation":"deterministic retrieval availability, not LLM answer quality"}
                contracts.append(record(store,"memory_benefit","cross_session_fact_recall",{"scope":"user:a","query":"language"},harness,repetition,recall))
            contracts.append(record(store,"memory_benefit","scope_isolation",{"scope":"user:b"},"memory-on",repetition,
                lambda:(not memory.search("user:b","language"), {"cross_scope_leaks":len(memory.search("user:b","language"))})))
    report = {"benchmark_version":1,"mode":"offline-contracts","harness_passed":harness_passed,"records":contracts,
        "memory_comparison":compare_paired([r for r in contracts if r["case_id"] == "cross_session_fact_recall"],"memory-off","memory-on"),
        "limitations":"No live LLM trials. Retrieval availability is not evidence of semantic memory benefit. Do not aggregate these layers into one score."}
    store.write("layers.json",report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts",default=".eval")
    parser.add_argument("--repetitions",type=int,default=1)
    args = parser.parse_args()
    store = ArtifactStore(args.artifacts,{"benchmark_version":1,"mode":"offline-contracts"})
    report = asyncio.run(run_layers(store,args.repetitions))
    print(json.dumps({"artifact_dir":str(store.root),"memory_comparison":report["memory_comparison"]},ensure_ascii=False,indent=2))
    return int(not report["harness_passed"] or any(r["outcome"] == "errored" or (r["harness_id"] != "memory-off" and not r["scores"].get("correctness")) for r in report["records"]))


if __name__ == "__main__":
    raise SystemExit(main())
