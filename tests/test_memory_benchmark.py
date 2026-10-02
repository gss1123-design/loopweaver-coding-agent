import asyncio
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/"src")]
import pytest
from evals.artifacts import ArtifactStore
from evals.memory_benchmark import CASES,run_memory_benchmark
from coding_agent.memory import MemoryStore


def test_memory_on_off_agent_runs_are_paired_and_keep_evidence(tmp_path):
    store = ArtifactStore(tmp_path,{"mode":"test-memory"})
    report = asyncio.run(run_memory_benchmark(store,2))
    assert report["passed"]
    assert len(report["records"]) == len(CASES)*4
    comparisons = {r["case_id"]:r for r in report["comparisons"]}
    assert comparisons["preference_hit"]["tool_calls_mean_delta"] == -1
    assert comparisons["cold_miss"]["tool_calls_mean_delta"] == 1
    for comparison in comparisons.values():
        assert comparison["paired_runs"] == 2
        assert comparison["tokens"]["mean_delta"] is None
        assert comparison["cost"]["mean_delta"] is None
    records = [json.loads(line) for line in (store.root/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
    for record in records:
        assert (store.root/record["artifact_dir"]/"events.jsonl").exists()
        assert (store.root/record["artifact_dir"]/"session.jsonl").exists()
    for record in report["records"]:
        if record["case_id"] in {"expired_fact","deleted_fact","scope_isolation","verify_stale_fact"}:
            assert record["evidence"]["source_reads"] == 1


def test_invalid_repetitions_refused(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(run_memory_benchmark(ArtifactStore(tmp_path,{}),0))


def test_updated_fact_and_delete_are_audited(tmp_path):
    memory=MemoryStore(tmp_path)
    memory.put("workspace","language","Java",kind="preference",source="user:1")
    memory.put("workspace","language","Python",kind="preference",source="user:2")
    fresh=MemoryStore(tmp_path)
    assert fresh.search("workspace","language")[0]["value"] == "Python"
    fresh.delete("workspace","language",source="user:3")
    assert not memory.search("workspace","language")
    with memory.connect() as db:
        rows=db.execute("SELECT action,source FROM memory_audit ORDER BY rowid").fetchall()
    assert [(r["action"],r["source"]) for r in rows] == [("upsert","user:1"),("upsert","user:2"),("delete","user:3")]


def test_irrelevant_queries_and_budget_do_not_inject_all_memory(tmp_path):
    memory=MemoryStore(tmp_path)
    for i in range(8):
        memory.put("workspace",f"language{i}","Python",kind="fact",source="fixture")
    assert not memory.search("workspace","unrelated")
    result=memory.search("workspace","language",limit=2,budget=500)
    assert len(result) <= 2
    assert sum(len(json.dumps(item,ensure_ascii=False)) for item in result) <= 500
    assert not memory.search("workspace","language",budget=1)
