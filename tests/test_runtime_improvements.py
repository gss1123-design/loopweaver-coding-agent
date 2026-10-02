import asyncio
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pytest
from ai.models import get_model
from ai.types import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from coding_agent.agent_session import AgentSession
from coding_agent.types import AgentSessionOptions
from coding_agent.recovery import batch_key, recover_completed_batch
from coding_agent.serde import message_to_dict
from evals.artifacts import ArtifactStore, compare_paired
from evals.runner import run_suite
from coding_agent.memory import MemoryStore
from coding_agent.factory import create_agent_session
from coding_agent.types import CreateAgentSessionOptions
from coding_agent.sandbox import docker_command, sandbox_tools
from coding_agent.builtin_tools import create_builtin_tools


def test_artifacts_are_isolated_and_contain_replay_inputs(tmp_path):
    first = ArtifactStore(tmp_path, {"provider": "scripted"})
    second = ArtifactStore(tmp_path, {"provider": "scripted"})
    assert first.root != second.root
    assert all(r.passed for r in asyncio.run(run_suite(artifacts=first, repetitions=2)))
    records = [json.loads(line) for line in (first.root / "runs.jsonl").read_text().splitlines()]
    assert len(records) == 4
    for record in records:
        assert (first.root / record["artifact_dir"] / "events.jsonl").exists()
        assert (first.root / record["artifact_dir"] / "session.jsonl").exists()
        assert record["telemetry"]["cost"] is None


def test_missing_measurement_never_counts_as_zero():
    base = dict(layer="memory", case_id="x", input_hash="h", repetition=1,
                outcome="scored", scores={"correctness": 1}, telemetry={"tokens": None})
    result = compare_paired([{**base, "harness_id": "base"}, {**base, "harness_id": "candidate"}], "base", "candidate")
    assert result["paired_runs"] == 1
    assert result["tokens"]["mean_delta"] is None
    assert result["tokens"]["eligible_pairs"] == 0


def test_partial_batch_blocks_even_with_tool_result_tail(tmp_path):
    session = AgentSession(AgentSessionOptions(model=get_model("openai-standard", "gpt-4o-mini"), workspace_dir=tmp_path))
    assistant = AssistantMessage(content=[ToolCall(id="a", name="write"), ToolCall(id="b", name="bash")], stop_reason="toolUse")
    session.agent.set_messages([UserMessage(content="task"), assistant, ToolResultMessage(tool_call_id="a", tool_name="write")])
    with patch.object(session, "continue_run", new=AsyncMock()) as continued:
        with pytest.raises(ValueError, match="副作用"):
            asyncio.run(session.resume_run())
        continued.assert_not_awaited()
    session.close()


def test_committed_result_recovers_without_reexecuting(tmp_path):
    session = AgentSession(AgentSessionOptions(model=get_model("openai-standard", "gpt-4o-mini"), workspace_dir=tmp_path))
    assistant = AssistantMessage(content=[ToolCall(id="a", name="write")], stop_reason="toolUse")
    result = ToolResultMessage(tool_call_id="a", tool_name="write")
    async def commit():
        await session._on_agent_event({"type": "message_end", "message": assistant})
        await session._on_agent_event({"type": "message_end", "message": result})
    asyncio.run(commit())
    session.agent.set_messages([assistant])  # simulate missing context projection
    with patch.object(session, "continue_run", new=AsyncMock(return_value=[])) as continued:
        asyncio.run(session.resume_run())
        continued.assert_awaited_once()
        assert session.messages[-1].tool_call_id == "a"
    session.close()


def test_results_from_other_batch_do_not_recover():
    assistant = AssistantMessage(content=[ToolCall(id="a", name="write")], stop_reason="toolUse")
    other = AssistantMessage(content=[ToolCall(id="a", name="write", arguments={"path": "other"})], stop_reason="toolUse")
    journal = [{"kind": "tool_result_committed", "payload": {"batch_key": batch_key(other), "message": message_to_dict(ToolResultMessage(tool_call_id="a", tool_name="write"))}}]
    with pytest.raises(ValueError, match="副作用"):
        recover_completed_batch(assistant, [], journal)


def test_memory_scopes_sources_and_expiry(tmp_path):
    store = MemoryStore(tmp_path)
    store.put("a", "language", "Python", kind="preference", source="user:1")
    store.put("b", "language", "Java", kind="preference", source="user:2")
    assert store.search("a", "language")[0]["value"] == "Python"
    with patch("coding_agent.memory.time.time", return_value=1):
        store.put("a", "expired", "old", kind="fact", source="task:1", ttl_seconds=1)
    assert not store.search("a", "expired")
    store.delete("a", "language", source="user:3")
    assert not store.search("a", "language")
    assert store.search("b", "language")
    with pytest.raises(ValueError, match="credentials"):
        store.put("a", "secret", "api_key=abc", kind="fact", source="user")


def test_memory_refreshes_without_prompt_accumulation(tmp_path):
    memory = tmp_path / ".xingclaw" / "MEMORY.md"
    memory.parent.mkdir()
    memory.write_text("first", encoding="utf-8")
    session = create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path, model=get_model("openai-standard", "gpt-4o-mini"), load_workspace_resources=False))
    assert "first" in session._with_extra_system_prompt(None)
    memory.write_text("second", encoding="utf-8")
    prompt = session._with_extra_system_prompt(None)
    assert "second" in prompt and "first" not in prompt
    session.close()


def test_sandbox_command_is_restricted(tmp_path):
    command = docker_command(tmp_path, "xingclaw-sandbox:local", "test-name", read_only=True)
    assert command[command.index("--network")+1] == "none"
    assert "--read-only" in command
    assert command[command.index("--mount")+1].endswith(",readonly")
    assert "--privileged" not in command


def test_real_docker_read_write_and_boundary(tmp_path):
    # Opt in to integration testing; ordinary test runs need no Docker daemon.
    import os
    if os.environ.get("XINGCLAW_TEST_DOCKER") != "1":
        pytest.skip("Set XINGCLAW_TEST_DOCKER=1 for container integration")
    tools = sandbox_tools(create_builtin_tools(tmp_path, ["read", "write"]), tmp_path, "xingclaw-sandbox:local", {})
    by_name = {t.name: t for t in tools}
    async def check():
        await by_name["write"].execute("w", {"path":"hello.txt", "content":"sandbox-ok"})
        result = await by_name["read"].execute("r", {"path":"hello.txt"})
        assert "sandbox-ok" in result.content[0].text
        with pytest.raises(RuntimeError, match="boundary"):
            await by_name["read"].execute("escape", {"path":"../escape.txt"})
    asyncio.run(check())
    assert (tmp_path / "hello.txt").read_text() == "sandbox-ok"


def test_checkpoint_failure_prevents_tool_execution():
    from agent_core import AgentTool, AgentToolResult
    from agent_core.agent_loop import _PreparedToolCall, _execute_prepared_tool_call
    execute = AsyncMock(return_value=AgentToolResult(content=[]))
    tool = AgentTool(name="write",label="write",description="",parameters={},execute=execute)
    prepared = _PreparedToolCall(tool_call=ToolCall(id="a",name="write"),tool=tool,args={})
    async def fail(event):
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(_execute_prepared_tool_call(prepared,fail,None))
    execute.assert_not_awaited()


def test_subagent_parallel_limit_order_and_chain(tmp_path):
    from ai.types import TextContent
    session = AgentSession(AgentSessionOptions(model=get_model("openai-standard", "gpt-4o-mini"),workspace_dir=tmp_path))
    active = 0
    peak = 0
    received = []
    closed = []
    class Lane:
        last_trace = {"status":"completed"}
        def __init__(self, index):
            self.session_id = f"s_lane_{index}"
        async def prompt(self, text, *, extra_system_prompt=None):
            nonlocal active,peak
            assert "Do not modify files" in extra_system_prompt
            received.append(text)
            active += 1
            peak = max(peak,active)
            try:
                await asyncio.sleep(0.01)
                return [AssistantMessage(content=[TextContent(text=text.splitlines()[0])])]
            finally:
                active -= 1
    count = 0
    def create(name, *, read_only=False):
        nonlocal count
        assert read_only
        count += 1
        lane = Lane(count)
        session._lanes[str(count)] = lane
        return lane
    def close(lane_id):
        closed.append(lane_id)
        session._lanes.pop(lane_id)
    tool = session._build_subagent_tool()
    with patch.object(session,"create_lane",side_effect=create),patch.object(session,"close_lane",side_effect=close):
        tasks = [{"task":f"task{i}","role":"reviewer"} for i in range(8)]
        result = asyncio.run(tool.execute("batch",{"mode":"parallel","tasks":tasks}))
        assert peak == 4 and len(closed) == 8
        assert [child["role"] for child in result.details["children"]] == ["reviewer"]*8
        assert result.content[0].text.index("task0") < result.content[0].text.index("task7")
        received.clear()
        asyncio.run(tool.execute("chain",{"mode":"chain","tasks":[{"task":"first"},{"task":"second"}]}))
        assert "Previous step output" in received[1] and "first" in received[1]
        assert not session._lanes
    session.close()


def test_four_layer_benchmark_and_unique_repetitions(tmp_path):
    from evals.layers import run_layers
    store = ArtifactStore(tmp_path,{"mode":"contracts"})
    report = asyncio.run(run_layers(store,repetitions=2))
    assert report["harness_passed"]
    records = [json.loads(line) for line in (store.root/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["layer"] for r in records} == {"harness_regression","context_governance","recovery_correctness","memory_benefit"}
    harness = [r for r in records if r["layer"] == "harness_regression"]
    assert len({(r["case_id"],r["repetition"]) for r in harness}) == 4
    assert report["memory_comparison"]["paired_runs"] == 2


def test_new_batch_intent_cannot_reuse_old_result():
    assistant = AssistantMessage(content=[ToolCall(id="a",name="write")],stop_reason="toolUse")
    key = batch_key(assistant)
    marker = {"kind":"tool_batch_committed","payload":{"batch_key":key}}
    result = {"kind":"tool_result_committed","payload":{"batch_key":key,"message":message_to_dict(ToolResultMessage(tool_call_id="a",tool_name="write"))}}
    with pytest.raises(ValueError,match="副作用"):
        recover_completed_batch(assistant,[],[marker,result,marker])
