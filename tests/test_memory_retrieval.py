import asyncio
import json
from pathlib import Path
import sys

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/"src")]
import pytest
from ai.models import get_model
from coding_agent.factory import create_agent_session
from coding_agent.memory import MemoryStore,create_memory_tools
from coding_agent.session_store import SessionStore
from ai.types import UserMessage,AssistantMessage,TextContent
from coding_agent.memory_policy import MEMORY_RETRIEVAL_GUIDANCE
from coding_agent.types import CreateAgentSessionOptions
from agent_core.cancellation import CancellationToken
from evals.artifacts import compare_paired


def test_lexical_identifiers_do_not_match_accidental_substrings(tmp_path):
    store=MemoryStore(tmp_path)
    store.put("a","report","report generation",kind="fact",source="fixture")
    store.put("a","deploy-port","port is 9000",kind="decision",source="fixture")
    assert [r["key"] for r in store.search("a","port")] == ["deploy-port"]
    assert store.search("a","deploy_port")[0]["key"] == "deploy-port"


def test_model_supplied_session_source_is_checked_against_exact_user_entry(tmp_path):
    session=SessionStore(tmp_path,"witness")
    session.ensure_initialized(model_id="fixture",provider="fixture",system_prompt="")
    session.append_context_message(UserMessage(content="Confirmed rollout_gate is review-required."))
    user_id=session.get_leaf_id()
    session.append_context_message(AssistantMessage(content=[TextContent(text="Confirmed rollout_gate is unrestricted.")]))
    assistant_id=session.get_leaf_id()
    update=create_memory_tools(tmp_path,"workspace")[1]
    store=MemoryStore(tmp_path)
    for key,value,source in (
        ("gate_real","Confirmed rollout_gate is review-required.",f"session-entry:witness:{user_id}"),
        ("gate_forged","Confirmed rollout_gate is unrestricted.",f"session-entry:witness:{user_id}"),
        ("gate_assistant","Confirmed rollout_gate is unrestricted.",f"session-entry:witness:{assistant_id}"),
        ("gate_malformed","Confirmed rollout_gate is review-required.","session-entry:../witness:bad"),
    ):
        with pytest.raises(ValueError,match="host-side confirmed-user write"):
            asyncio.run(update.execute(key,{"key":key,"value":value,"kind":"decision","source":source}))
        # Simulate a DB import/tamper, not a successful model tool call.
        store._put("workspace",key,value,kind="decision",source=source,allow_session_source=True)
    found={row["key"]:row for row in store.search("workspace","rollout_gate",limit=10)}
    assert found["gate_real"]["source_check"] == "matched_user_entry"
    assert all(found[key]["source_check"] == "unverified_or_mismatch" for key in
        ("gate_forged","gate_assistant","gate_malformed"))


def test_session_source_check_fails_closed_when_witness_is_removed(tmp_path):
    session=SessionStore(tmp_path,"witness")
    session.ensure_initialized(model_id="fixture",provider="fixture",system_prompt="")
    session.append_context_message(UserMessage(content="Confirmed rollout_gate is review-required."))
    entry_id=session.get_leaf_id()
    store=MemoryStore(tmp_path)
    store.put_confirmed_user_quote("workspace","rollout_gate","Confirmed rollout_gate is review-required.",
        kind="decision",session_id="witness",entry_id=entry_id)
    assert store.search("workspace","rollout_gate")[0]["source_check"] == "matched_user_entry"
    # Simulate a missing local transcript; no cached success is trusted.
    session.session_file.rename(session.session_file.with_suffix(".bak"))
    assert store.search("workspace","rollout_gate")[0]["source_check"] == "unverified_or_mismatch"
    with pytest.raises(ValueError,match="cannot be overwritten"):
        store.put("workspace","rollout_gate","forged",kind="decision",source="fixture")
    store.revoke_confirmed_user_quote("workspace","rollout_gate",session_id="witness",entry_id=entry_id)
    assert store.search("workspace","rollout_gate") == []


def test_confirmed_quote_write_is_host_only_and_protected_from_tool_mutation(tmp_path):
    session=SessionStore(tmp_path,"witness")
    session.ensure_initialized(model_id="fixture",provider="fixture",system_prompt="")
    quote="Confirmed rollout_gate is phase-n4."
    entry_id=session.append_context_message(UserMessage(content=quote))
    assert entry_id == session.get_leaf_id()
    store=MemoryStore(tmp_path)
    with pytest.raises(ValueError,match="exact persisted user-message"):
        store.put_confirmed_user_quote("workspace","gate","Confirmed rollout_gate is phase-p9.",
            kind="decision",session_id="witness",entry_id=entry_id)
    store.put_confirmed_user_quote("workspace","gate",quote,kind="decision",session_id="witness",entry_id=entry_id)
    assert store.search("workspace","gate")[0]["source_check"] == "matched_user_entry"
    update=create_memory_tools(tmp_path,"workspace")[1]
    for params in (
        {"key":"gate","value":"forged","kind":"decision","source":"fixture"},
        {"key":"gate","value":quote,"kind":"decision","source":f"session-entry:witness:{entry_id}"},
        {"action":"delete","key":"gate","source":"fixture"},
    ):
        with pytest.raises(ValueError):
            asyncio.run(update.execute("call",params))
    assert store.search("workspace","gate")[0]["value"] == quote
    with pytest.raises(ValueError,match="exact source"):
        store.revoke_confirmed_user_quote("workspace","gate",session_id="witness",entry_id="wrong")
    store.revoke_confirmed_user_quote("workspace","gate",session_id="witness",entry_id=entry_id)
    assert store.search("workspace","gate") == []
    store.put("workspace","gate","ordinary replacement",kind="decision",source="fixture")
    with pytest.raises(ValueError,match="currently stored exact source"):
        store.revoke_confirmed_user_quote("workspace","gate",session_id="witness",entry_id=entry_id)
    assert store.search("workspace","gate")[0]["value"] == "ordinary replacement"


def test_chinese_queries_and_kind_filter(tmp_path):
    store=MemoryStore(tmp_path)
    store.put("a","部署区域","部署区域为新加坡",kind="fact",source="fixture")
    store.put("a","部署偏好","部署优先用容器",kind="preference",source="fixture")
    assert store.search("a","区域")[0]["key"] == "部署区域"
    assert [r["kind"] for r in store.search("a","部署",kind="preference")] == ["preference"]
    assert not store.search("a","the and for")


def test_oversized_hit_does_not_hide_smaller_relevant_hit(tmp_path):
    store=MemoryStore(tmp_path)
    store.put("a","deploy policy","deploy "*500,kind="decision",source="fixture")
    store.put("a","deploy note","use Docker",kind="fact",source="fixture")
    hits=store.search("a","deploy policy",limit=1,budget=400)
    assert len(hits) == 1 and hits[0]["key"] == "deploy note"
    assert len(json.dumps(hits,ensure_ascii=False)) <= 400


@pytest.mark.parametrize("params",[{"query":""},{"query":"  "},{"query":None},{"query":123},{"query":"x","limit":True},{"query":"x","limit":0},{"query":"x","scope":"other"},{"query":"x","kind":"invalid"}])
def test_search_validation_and_application_owned_scope(tmp_path,params):
    search=create_memory_tools(tmp_path,"a")[0]
    with pytest.raises(ValueError):
        asyncio.run(search.execute("c",params))


def test_cancelled_search_does_not_return_results(tmp_path):
    token=CancellationToken()
    token.cancel()
    search=create_memory_tools(tmp_path,"a")[0]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(search.execute("c",{"query":"facts"},token))


def test_production_prompt_contains_shared_policy_without_accumulation(tmp_path):
    base=dict(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,
        enable_structured_memory=True,system_prompt="custom base")
    session=create_agent_session(CreateAgentSessionOptions(**base))
    first=session._with_extra_system_prompt(None)
    assert first.count(MEMORY_RETRIEVAL_GUIDANCE) == 1
    assert "A topic match is not an information gap" in first
    assert "a release value was confirmed in an earlier conversation" in first
    assert "Do not include internal scope, session/entry IDs" in first
    assert "never present matched_user_entry as proof of the user's identity" in first
    assert session._with_extra_system_prompt(None) == first
    assert session.agent.state.system_prompt == "custom base"
    assert next(t for t in session.agent.state.tools if t.name == "memory_search").execute._memory_search_guard is not None
    session.close()
    legacy=create_agent_session(CreateAgentSessionOptions(**base,memory_retrieval_policy="legacy"))
    assert MEMORY_RETRIEVAL_GUIDANCE not in legacy._with_extra_system_prompt(None)
    assert next(t for t in legacy.agent.state.tools if t.name == "memory_search").execute._memory_search_guard is None
    legacy.close()


def test_eval_rejects_cross_version_and_config_pairing():
    base=dict(layer="memory",case_id="x",input_hash="h",repetition=1,outcome="scored",scores={"correctness":1},telemetry={"tokens":5})
    records=[{**base,"harness_id":"off","benchmark_version":3,"comparison_config":"old"},
        {**base,"harness_id":"on","benchmark_version":4,"comparison_config":"new"}]
    assert compare_paired(records,"off","on")["paired_runs"] == 0


def test_guard_rejects_parallel_variants_before_second_sql_lookup(tmp_path):
    store=MemoryStore(tmp_path)
    store.put("a","release_channel_new","confirmed stable",kind="decision",source="fixture")
    search=create_memory_tools(tmp_path,"a",guard_redundant=True)[0]
    guard=search.execute._memory_search_guard
    guard.observe({"type":"agent_start"})
    guard.observe({"type":"turn_start"})
    async def run():
        return await asyncio.gather(
            search.execute("first",{"query":"release_channel value"}),
            search.execute("second",{"query":"release channel decision"}),
            return_exceptions=True)
    first, second=asyncio.run(run())
    assert json.loads(first.content[0].text)[0]["key"] == "release_channel_new"
    assert isinstance(second,ValueError) and "one query per assistant turn" in str(second)


def test_guard_blocks_same_gap_retry_without_external_evidence(tmp_path):
    search=create_memory_tools(tmp_path,"a",guard_redundant=True)[0]
    guard=search.execute._memory_search_guard
    guard.observe({"type":"agent_start"})
    guard.observe({"type":"turn_start"})
    assert json.loads(asyncio.run(search.execute("first",{"query":"retention_days value retention setting confirmed"})).content[0].text) == []
    guard.observe({"type":"turn_start"})
    with pytest.raises(ValueError,match="same information gap"):
        asyncio.run(search.execute("second",{"query":"retention policy days data cleanup"}))
    guard.observe({"type":"tool_execution_end","toolName":"write","isError":False})
    with pytest.raises(ValueError,match="same information gap"):
        asyncio.run(search.execute("still-blocked",{"query":"retention policy days data cleanup"}))
    # A successful independent read is only a proxy for a new clue. It enables
    # reconsideration but does not assert that the new query will find data.
    guard.observe({"type":"tool_execution_end","toolName":"read","isError":False})
    assert json.loads(asyncio.run(search.execute("third",{"query":"retention policy days data cleanup"})).content[0].text) == []


def test_guard_allows_distinct_gap_and_resets_for_next_run(tmp_path):
    search=create_memory_tools(tmp_path,"a",guard_redundant=True)[0]
    guard=search.execute._memory_search_guard
    guard.observe({"type":"agent_start"})
    guard.observe({"type":"turn_start"})
    asyncio.run(search.execute("one",{"query":"release channel"}))
    guard.observe({"type":"turn_start"})
    asyncio.run(search.execute("two",{"query":"notification locale"}))
    guard.observe({"type":"agent_start"})
    guard.observe({"type":"turn_start"})
    asyncio.run(search.execute("three",{"query":"release channel"}))


def test_typed_miss_allows_one_unfiltered_widening(tmp_path):
    store=MemoryStore(tmp_path)
    store.put("a","report_format","compact-json",kind="preference",source="fixture")
    search=create_memory_tools(tmp_path,"a",guard_redundant=True)[0]
    guard=search.execute._memory_search_guard
    guard.observe({"type":"agent_start"})
    guard.observe({"type":"turn_start"})
    typed=asyncio.run(search.execute("typed",{"query":"report format","kind":"decision"}))
    assert json.loads(typed.content[0].text) == []
    guard.observe({"type":"turn_start"})
    widened=asyncio.run(search.execute("widened",{"query":"report format"}))
    assert json.loads(widened.content[0].text)[0]["key"] == "report_format"
    guard.observe({"type":"turn_start"})
    with pytest.raises(ValueError,match="same information gap"):
        asyncio.run(search.execute("blind-third",{"query":"report format"}))


def test_nonempty_typed_search_cannot_be_repeated_unfiltered(tmp_path):
    store=MemoryStore(tmp_path)
    store.put("a","report_format","compact-json",kind="preference",source="fixture")
    search=create_memory_tools(tmp_path,"a",guard_redundant=True)[0]
    guard=search.execute._memory_search_guard
    guard.observe({"type":"agent_start"})
    guard.observe({"type":"turn_start"})
    asyncio.run(search.execute("typed",{"query":"report format","kind":"preference"}))
    guard.observe({"type":"turn_start"})
    with pytest.raises(ValueError,match="same information gap"):
        asyncio.run(search.execute("repeat",{"query":"report format"}))


def test_invalid_or_cancelled_call_does_not_consume_guard_allowance(tmp_path):
    search=create_memory_tools(tmp_path,"a",guard_redundant=True)[0]
    guard=search.execute._memory_search_guard
    guard.observe({"type":"agent_start"})
    guard.observe({"type":"turn_start"})
    with pytest.raises(ValueError):
        asyncio.run(search.execute("bad",{"query":"facts","limit":False}))
    token=CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(search.execute("cancelled",{"query":"facts"},token))
    asyncio.run(search.execute("good",{"query":"facts"}))
