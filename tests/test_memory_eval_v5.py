import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/"src")]
from evals.live_memory import TASKS,HISTORY_TASK,VERIFIED_HISTORY_TASK,select_tasks,task_evidence,reference_context,run_arm,run_suite,build_report,abstention_scores,build_system_prompt,seed_witness,hide_source_check
from evals.memory_cases import EXTENDED_TASKS
from evals.memory_scale_cases import SCALED_TASKS,DENSITY_TASKS,WITNESSED_SPOOF_TASK,WITNESSED_SPOOF_TASKS,NEUTRAL_ABLATION_TASKS
from evals.memory_audit import audit_retrieval
from evals.artifacts import ArtifactStore
from coding_agent.memory import MemoryStore,create_memory_tools
from coding_agent.memory_policy import HISTORICAL_EVIDENCE_GUIDANCE
from ai.models import get_model
from ai.types import AssistantMessage


ALL_TASKS = (*TASKS,HISTORY_TASK,VERIFIED_HISTORY_TASK,*EXTENDED_TASKS,*SCALED_TASKS,*DENSITY_TASKS,*WITNESSED_SPOOF_TASKS,*NEUTRAL_ABLATION_TASKS)


def call(cid,query="release channel",**args):
    return {"role":"assistant","content":[{"type":"toolCall","id":cid,"name":"memory_search","arguments":{"query":query,**args}}]}


def result(cid,rows,error=False):
    return {"role":"toolResult","tool_call_id":cid,"tool_name":"memory_search","is_error":error,"content":[{"type":"text","text":json.dumps(rows)}]}


def test_fixed_tasks_cover_five_categories_and_no_duplicate_keys():
    assert len(ALL_TASKS) == 22
    assert len({t.name for t in ALL_TASKS}) == 22
    assert {t.category for t in ALL_TASKS} == {"current_documents","historical_context","historical_context_verified","updated_facts","conflicting_memories","no_evidence","source_authenticity_unverified","source_authenticity_registered","freshness_equal_trust","source_authenticity_registered_density_16","source_authenticity_registered_density_160","source_id_spoof_witnessed"}
    assert all(t.retrieval_needed is None for t in ALL_TASKS if t.category == "no_evidence")
    for task in ALL_TASKS:
        rows = task_evidence(task)
        keys = {row["key"] for row in rows}
        assert len(keys) == len(rows)
        assert set(task.necessary_keys) <= keys and set(task.stale_keys) <= keys
        compile(task.source,task.name,"exec")
        for checks in (task.checks,task.abstention_checks,task.stale_checks,task.misattribution_checks):
            if checks:
                compile(checks,task.name,"exec")


def test_reference_baseline_contains_only_same_evidence_not_grader_annotations():
    for task in ALL_TASKS:
        context = reference_context(task)
        assert json.loads(context.split("\n",3)[-1]) == task_evidence(task)
        assert task.checks not in context
        assert "necessary_keys" not in context and "stale_checks" not in context


def test_scaled_controls_hold_other_dimensions_constant():
    unverified, registered, freshness = SCALED_TASKS
    rows = [task_evidence(task) for task in SCALED_TASKS]
    assert all(len(group) >= 70 for group in rows)
    assert rows[0][:72] == rows[1][:72] == rows[2][:72]
    assert len({row["updated"] for row in rows[1][-2:]}) == 1
    assert len({row["updated"] for row in rows[2][-2:]}) == 2
    assert "session:gate-417" in registered.notes
    assert all(row["source"] in freshness.notes for row in rows[2][-2:])
    assert "independent source registry" in unverified.notes
    for task in SCALED_TASKS:
        assert "review-required" not in task.prompt + task.notes + task.source


def test_density_cases_change_only_archive_count():
    reference = SCALED_TASKS[1]
    for task, count in zip(DENSITY_TASKS, (16,160)):
        assert len(task.memories) == count + 2
        assert task.memories[-2:] == reference.memories[-2:]
        assert (task.prompt,task.notes,task.source,task.checks,task.necessary_keys) == (
            reference.prompt,reference.notes,reference.source,reference.checks,reference.necessary_keys)


def test_witnessed_spoof_has_same_source_but_only_exact_user_quote_matches(tmp_path):
    task=WITNESSED_SPOOF_TASK
    seed_witness(task,tmp_path)
    store=MemoryStore(tmp_path)
    for row in task_evidence(task)[-2:]:
        store._put("workspace",row["key"],row["value"],kind=row["kind"],source=row["source"],allow_session_source=True)
    found={row["key"]:row for row in store.search("workspace","rollout_gate")}
    genuine,spoofed=task.memories[-2:]
    assert genuine.source == spoofed.source
    assert (genuine.key,spoofed.key) == ("rollout_gate_a","rollout_gate_b")
    assert "witness" not in genuine.key and "spoof" not in spoofed.key
    assert genuine.updated == spoofed.updated
    assert found[genuine.key]["source_check"] == "matched_user_entry"
    assert found[spoofed.key]["source_check"] == "unverified_or_mismatch"
    assert "review-required" not in task.prompt + task.notes + task.source
    assert reference_context(task).count("source_check") == 2


def test_witnessed_spoof_mirror_reverses_correct_row_position(tmp_path):
    original, mirror = WITNESSED_SPOOF_TASKS
    assert original.prompt == mirror.prompt and original.notes == mirror.notes
    assert original.witness_message == mirror.witness_message and original.checks == mirror.checks
    assert [m.key for m in original.memories[-2:]] == [m.key for m in mirror.memories[-2:]]
    for task, expected in ((original,"matched_user_entry"),(mirror,"unverified_or_mismatch")):
        workspace=tmp_path/task.name
        seed_witness(task,workspace)
        store=MemoryStore(workspace)
        for row in task_evidence(task)[-2:]:
            store._put("workspace",row["key"],row["value"],kind=row["kind"],source=row["source"],allow_session_source=True)
            with store.connect() as db:
                db.execute("UPDATE memories SET updated=? WHERE scope=? AND key=?",
                    (row["updated"],row["scope"],row["key"]))
        rows=store.search("workspace","rollout_gate")
        assert [row["key"] for row in rows] == ["rollout_gate_a","rollout_gate_b"]
        assert rows[0]["source_check"] == expected


def test_neutral_ablation_hides_only_check_field(tmp_path):
    task=NEUTRAL_ABLATION_TASKS[1]
    seed_witness(task,tmp_path)
    store=MemoryStore(tmp_path)
    for row in task_evidence(task)[-2:]:
        store._put("workspace",row["key"],row["value"],kind=row["kind"],source=row["source"],allow_session_source=True)
        with store.connect() as db:
            db.execute("UPDATE memories SET updated=? WHERE scope=? AND key=?",
                (row["updated"],row["scope"],row["key"]))
    checked=store.search("workspace","rollout_gate")
    tool=create_memory_tools(tmp_path,"workspace",guard_redundant=True)[0]
    guard=tool.execute._memory_search_guard
    hide_source_check(tool)
    assert tool.execute._memory_search_guard is guard
    unchecked=json.loads(asyncio.run(tool.execute("c",{"query":"rollout_gate"})).content[0].text)
    assert unchecked == [{k:v for k,v in row.items() if k != "source_check"} for row in checked]
    assert {row["value"] for row in checked} == {
        "Confirmed rollout_gate is phase-n4.","Confirmed rollout_gate is phase-p9."}


def test_only_tool_enabled_arm_mentions_memory_search():
    task=HISTORY_TASK
    off=build_system_prompt(task,"memory-off","selective")
    direct=build_system_prompt(task,"memory-context","selective")
    retrieval=build_system_prompt(task,"memory-on","selective")
    assert "memory_search" not in off and "memory_search" not in direct
    assert "memory_search" in retrieval
    assert HISTORICAL_EVIDENCE_GUIDANCE in direct and HISTORICAL_EVIDENCE_GUIDANCE in retrieval
    assert "harbor-47" not in off + retrieval and "harbor-47" in direct


def test_audit_distinguishes_empty_duplicate_error_and_stale():
    row = {"key":"old","value":"beta","source":"prior","updated":1}
    messages = [call("1"),result("1",[row]),call("2"," RELEASE   CHANNEL "),result("2",[row]),
        call("3",kind="fact"),result("3",[]),call("4","new decision"),result("4",[{"key":"new"}]),
        call("5","broken"),result("5",[],True),call("6","pending")]
    audit = audit_retrieval(messages,["new"],["old"])
    assert audit["search_count"] == 6
    assert audit["empty_searches"] == 1 and audit["duplicate_queries"] == 1
    assert audit["repeated_records"] == 1 and audit["unresolved_searches"] == 2
    assert audit["necessary_evidence_recall"] == 1
    assert audit["stale_keys_returned"] == ["old"]
    assert "stale_behavior" not in audit  # Returning evidence is not adopting it.


def test_unknown_or_invalid_results_are_not_empty_hits():
    messages = [call("1"),result("1",{"error":"not an array"})]
    audit = audit_retrieval(messages)
    assert audit["empty_searches"] == 0 and audit["unresolved_searches"] == 1
    assert audit["necessary_evidence_recall"] is None


def test_suppressed_tool_error_is_separate_from_empty_search():
    messages=[call("guarded"),{"role":"toolResult","tool_call_id":"guarded","tool_name":"memory_search",
        "is_error":True,"content":[{"type":"text","text":"Memory search suppressed: only one query per assistant turn"}]}]
    audit=audit_retrieval(messages)
    assert audit["suppressed_searches"] == 1
    assert audit["empty_searches"] == 0
    assert audit["unresolved_searches"] == 1


def test_unfinished_unchanged_stub_is_not_scored_as_model_invention():
    scores=abstention_scores(target_passed=False,abstention_passed=False,
        notes_unchanged=True,code_modified=False)
    assert scores == {"abstained":0}
    assert abstention_scores(target_passed=False,abstention_passed=False,
        notes_unchanged=True,code_modified=True)["no_invention"] == 0


def test_context_arm_has_no_memory_tool_and_seeded_arm_has_equal_data(tmp_path):
    captured=[]
    class FakeSession:
        def __init__(self):
            self.messages = [AssistantMessage()]
            self.agent = SimpleNamespace(state=SimpleNamespace(tools=[]))
        def subscribe(self,callback): return lambda:None
        async def prompt(self,prompt): pass
        def close(self): pass
    def create(options):
        captured.append(options)
        return FakeSession()
    baseline = {"passed":False,"exit_code":1,"stderr":"AssertionError"}
    passed = {"passed":True,"exit_code":0}
    store = ArtifactStore(tmp_path,{"offline":True})
    model = get_model("openai-standard","gpt-4o-mini")
    with patch("evals.live_memory.create_agent_session",side_effect=create),patch("evals.live_memory.grade",side_effect=[baseline,passed,baseline,baseline,passed,baseline]):
        context = asyncio.run(run_arm(HISTORY_TASK,False,1,store,model,"unused","selective","memory-context"))
        memory = asyncio.run(run_arm(HISTORY_TASK,True,1,store,model,"unused","selective","memory-on"))
    assert not captured[0].enable_structured_memory and captured[1].enable_structured_memory
    assert reference_context(HISTORY_TASK) in captured[0].system_prompt
    assert reference_context(HISTORY_TASK) not in captured[1].system_prompt
    rows = MemoryStore(memory["evidence"]["workspace"]).search("workspace","release")
    assert rows == task_evidence(HISTORY_TASK)
    assert context["input_hash"] == memory["input_hash"]
    report = build_report([context,memory])
    assert report["equal_information_comparison"]["paired_runs"] == 1
    assert report["arm_metrics"]["memory-context"]["correctness"]["mean"] == 1


def test_topk_context_arm_injects_only_selected_rows_without_memory_tool(tmp_path):
    captured=[]
    class FakeSession:
        messages = [AssistantMessage()]
        agent = SimpleNamespace(state=SimpleNamespace(tools=[]))
        def subscribe(self,callback): return lambda:None
        async def prompt(self,prompt): pass
        def close(self): pass
    def create(options):
        captured.append(options)
        return FakeSession()
    store=ArtifactStore(tmp_path,{"offline":True})
    model=get_model("openai-standard","gpt-4o-mini")
    baseline={"passed":False,"exit_code":1,"stderr":"AssertionError"}
    passed={"passed":True,"exit_code":0}
    with patch("evals.live_memory.create_agent_session",side_effect=create),patch("evals.live_memory.grade",side_effect=[baseline,passed]):
        record=asyncio.run(run_arm(TASKS[0],False,1,store,model,"unused","selective","memory-context-topk"))
    assert not captured[0].enable_structured_memory
    assert "Historical reference data" in captured[0].system_prompt
    assert "memory_search" not in captured[0].system_prompt
    assert record["evidence"]["topk_reference_keys"] == ["slug_policy"]
    assert build_report([record])["topk_context_comparison"]["paired_runs"] == 0


def test_three_arm_order_rotates_and_respects_selection(tmp_path):
    arms=[]
    async def arm(task,enabled,repetition,store,model,image,policy,name):
        arms.append((task.name,repetition,name))
        return {"layer":"live_memory_coding","case_id":task.name,"input_hash":"same","repetition":repetition,
            "harness_id":name,"outcome":"scored","scores":{"correctness":1},"telemetry":{"tokens":10,"tool_calls":0},"evidence":{"stop_reason":"stop"}}
    store=ArtifactStore(tmp_path,{"offline":True})
    with patch("evals.live_memory.run_arm",side_effect=arm):
        report=asyncio.run(run_suite(store,None,"unused",3,"selective",False,["historical_release"],True))
    assert len(arms) == 9
    assert [arms[i][2] for i in (0,3,6)] == ["memory-off","memory-on","memory-context"]
    assert report["equal_information_comparison"]["paired_runs"] == 3


def test_provenance_ablation_pairs_same_tasks_without_off_arm(tmp_path):
    seen=[]
    async def arm(task,enabled,repetition,store,model,image,policy,name):
        seen.append((task.name,name,enabled))
        return {"layer":"live_memory_coding","case_id":task.name,"input_hash":task.name,
            "repetition":repetition,"harness_id":name,"outcome":"scored",
            "scores":{"correctness":int(name == "memory-on")},
            "telemetry":{"tokens":10,"tool_calls":0},"evidence":{"stop_reason":"stop"}}
    store=ArtifactStore(tmp_path,{"offline":True})
    names=[task.name for task in NEUTRAL_ABLATION_TASKS]
    with patch("evals.live_memory.run_arm",side_effect=arm):
        report=asyncio.run(run_suite(store,None,"unused",1,"selective",False,names,False,True))
    assert len(seen) == 4
    assert all(enabled for _,_,enabled in seen)
    assert {name for _,name,_ in seen} == {"memory-on","memory-on-unchecked"}
    assert report["provenance_ablation_comparison"]["paired_runs"] == 2


def test_plan_mode_does_not_require_key_or_paid_permission():
    command=[sys.executable,"-m","evals.live_memory","--plan","--all-cases","--equal-context","--repetitions","3"]
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan["runs"] == len(ALL_TASKS)*3*3 and plan["api_calls"] == 0
    topk=subprocess.run([sys.executable,"-m","evals.live_memory","--plan","--case","slug_policy",
        "--equal-context","--topk-context"],capture_output=True,text=True,check=True)
    assert json.loads(topk.stdout)["runs"] == 4
