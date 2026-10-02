import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/"src")]
from evals.artifacts import ArtifactStore
from evals.analyze_memory import analyze


def test_trace_audit_counts_read_then_memory_and_real_usage(tmp_path):
    store=ArtifactStore(tmp_path,{})
    record={"case_id":"case","repetition":1,"harness_id":"memory-on","scores":{"correctness":1},"outcome":"scored"}
    messages=[
        {"role":"assistant","usage":{"input":10,"output":2},"content":[{"type":"toolCall","id":"r","name":"read","arguments":{"path":"PROJECT_NOTES.md"}}]},
        {"role":"toolResult","tool_call_id":"r","tool_name":"read","content":[{"type":"text","text":"notes"}]},
        {"role":"assistant","usage":{"input":20,"output":3},"content":[{"type":"toolCall","id":"m","name":"memory_search","arguments":{"query":"policy"}}]},
        {"role":"toolResult","tool_call_id":"m","tool_name":"memory_search","content":[{"type":"text","text":json.dumps([{"value":"same notes"}])}]},
    ]
    store.record(record,messages=messages)
    store.write("live-memory-report.json",{"records":[record]})
    result=analyze(store.root/"live-memory-report.json")
    totals=result["totals"]["memory-on"]
    assert totals["input_tokens"] == 30
    assert totals["output_tokens"] == 5
    assert totals["nonempty_searches"] == 1
    assert totals["searches_after_notes"] == 1


def test_trace_audit_includes_direct_context_arm(tmp_path):
    store=ArtifactStore(tmp_path,{})
    record={"case_id":"case","repetition":1,"harness_id":"memory-context","scores":{"correctness":1},"outcome":"scored"}
    store.record(record,messages=[{"role":"assistant","usage":{"input":40,"output":5},"content":[]}])
    store.write("live-memory-report.json",{"records":[record]})
    result=analyze(store.root/"live-memory-report.json")
    assert result["totals"]["memory-context"]["input_tokens"] == 40
    assert result["totals"]["memory-context"]["searches"] == 0
    assert result["observations"][0]["retrieval_audit"]["necessary_evidence_recall"] is None
