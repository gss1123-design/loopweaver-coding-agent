import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/"src")]
from evals.live_memory import TASKS,grade,build_report,fixture_guard,memory_guidance,HISTORY_TASK,VERIFIED_HISTORY_TASK,select_tasks,followthrough_diagnostics,seed_witness,task_evidence,topk_reference_context
from coding_agent.memory import MemoryStore
from types import SimpleNamespace


def test_grading_uses_restricted_docker_and_cleans_up(tmp_path):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,0,stdout="",stderr="")
    with patch("evals.live_memory.subprocess.run",side_effect=run):
        assert grade(tmp_path,TASKS[0],"test-image")["passed"]
    command=calls[0]
    assert command[command.index("--network")+1] == "none"
    assert command[command.index("--user")+1] == "65534:65534"
    assert command[command.index("--mount")+1].endswith(",readonly")
    assert "--read-only" in command and "-e" not in command
    assert calls[1][:3] == ["docker","rm","-f"]


def test_verifier_timeout_removes_named_container(tmp_path):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command,30)
        return subprocess.CompletedProcess(command,0,stdout="",stderr="")
    with patch("evals.live_memory.subprocess.run",side_effect=run):
        assert not grade(tmp_path,TASKS[0],"test-image")["passed"]
    assert calls[1][-1] == calls[0][calls[0].index("--name")+1]


def test_missing_live_cost_is_not_zero():
    record={"layer":"live_memory_coding","case_id":"slug_policy","input_hash":"h","harness_id":"memory-off", "repetition":1,
        "outcome":"scored","scores":{"correctness":0},"telemetry":{"cost":None}}
    report=build_report([record])
    assert report["cost_upper_estimate_usd"] is None
    assert report["comparison"]["cost"]["mean_delta"] is None
    assert report["comparison"]["diagnostics"]


def test_fixture_guard_blocks_runtime_logs_and_notes_writes():
    for name,path in [("read",".loopweaver/sessions/log"),("write","PROJECT_NOTES.md"),("read","../app.py")]:
        ctx=SimpleNamespace(tool_call=SimpleNamespace(name=name),args={"path":path})
        assert fixture_guard(ctx).block
    for name,path in [("read","app.py"),("read","PROJECT_NOTES.md"),("edit","app.py")]:
        assert fixture_guard(SimpleNamespace(tool_call=SimpleNamespace(name=name),args={"path":path})) is None


def test_selective_guidance_and_history_evidence_separation():
    assert "ONLY" in memory_guidance("selective")
    assert "skip memory_search" in memory_guidance("selective")
    assert "harbor-47" in HISTORY_TASK.memory
    assert "harbor-47" not in HISTORY_TASK.prompt+HISTORY_TASK.notes+HISTORY_TASK.source


def test_explicit_selection_limits_paid_cases():
    assert select_tasks(["historical_release", "historical_release"]) == (HISTORY_TASK,)
    assert select_tasks(["verified_historical_release"]) == (VERIFIED_HISTORY_TASK,)
    assert select_tasks(None, False) == TASKS
    assert len(select_tasks(None, True)) == 4


def test_verified_historical_fixture_has_matching_user_entry(tmp_path):
    task = VERIFIED_HISTORY_TASK
    seed_witness(task, tmp_path)
    item = task_evidence(task)[0]
    store = MemoryStore(tmp_path)
    store.put_confirmed_user_quote(item["scope"], item["key"], item["value"],
        kind=item["kind"], session_id=task.witness_session_id, entry_id=task.witness_entry_id)
    assert store.search("workspace", "release_policy")[0]["source_check"] == "matched_user_entry"


def test_topk_baseline_uses_user_prompt_and_same_retriever_without_gold_keys(tmp_path):
    task = VERIFIED_HISTORY_TASK
    seed_witness(task, tmp_path)
    item = task_evidence(task)[0]
    store = MemoryStore(tmp_path)
    store.put_confirmed_user_quote(item["scope"], item["key"], item["value"],
        kind=item["kind"], session_id=task.witness_session_id, entry_id=task.witness_entry_id)
    context, keys = topk_reference_context(store,task)
    assert keys == ["verified_historical_release"]
    assert json.loads(context.split("\n",3)[-1])[0]["source_check"] == "matched_user_entry"


def test_abstention_is_separate_from_target_completion():
    record = {"layer":"live_memory_coding", "case_id":"historical_release",
        "input_hash":"h", "harness_id":"memory-off", "repetition":1,
        "outcome":"scored", "scores":{"correctness":0,"abstained":1,"no_invention":1},
        "telemetry":{"cost":None}}
    report = build_report([record])
    metrics = report["arm_metrics"]
    assert metrics["memory-off"]["correctness"]["mean"] == 0
    assert metrics["memory-off"]["no_invention"]["mean"] == 1
    assert metrics["memory-on"]["no_invention"]["mean"] is None


def test_retrieval_hit_without_edit_is_not_task_followthrough():
    assert followthrough_diagnostics(tool_names=["read", "memory_search"], code_modified=False) == {
        "code_modified":False, "retrieved_but_no_edit":True}
    assert followthrough_diagnostics(tool_names=["read", "memory_search", "write"], code_modified=True) == {
        "code_modified":True, "retrieved_but_no_edit":False}
    assert not followthrough_diagnostics(tool_names=["read"], code_modified=False)["retrieved_but_no_edit"]
