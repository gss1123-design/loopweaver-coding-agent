import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from ai.models import get_model
from ai.types import AssistantMessage, TextContent
from coding_agent.agent_session import AgentSession
from coding_agent.factory import create_agent_session
from coding_agent.types import CreateAgentSessionOptions
from coding_agent.workspaces import WorkspaceSnapshot, files
from coding_agent.sandbox import sandbox_tools
from coding_agent.builtin_tools import create_builtin_tools


def test_snapshot_filters_secrets_state_and_dependencies(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"app.py").write_text("original")
    (source/".env").write_text("secret")
    (source/"private.key").write_text("secret")
    (source/".loopweaver").mkdir()
    (source/".loopweaver"/"journal.jsonl").write_text("state")
    snapshot = WorkspaceSnapshot(source,tmp_path/"candidate")
    assert list(snapshot.base) == ["app.py"]
    assert not (snapshot.target/".env").exists()
    (snapshot.target/"app.py").write_text("changed")
    (snapshot.target/".env").write_text("attempted")
    assert snapshot.publish() == ["app.py"]
    assert (source/".env").read_text() == "secret"


def test_conflicts_block_entire_merge_and_nonconflicting_workers_merge(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    (source/"a").write_text("base")
    (source/"b").write_text("base")
    first=WorkspaceSnapshot(source,tmp_path/"first")
    second=WorkspaceSnapshot(source,tmp_path/"second")
    (first.target/"a").write_text("first")
    first.publish()
    (second.target/"a").write_text("second")
    (second.target/"b").write_text("second")
    with pytest.raises(ValueError,match="conflict"):
        second.publish()
    assert (source/"b").read_text() == "base"
    (second.target/"a").write_text("base")
    second.publish()
    assert (source/"a").read_text() == "first" and (source/"b").read_text() == "second"


def test_real_worker_rebinds_tools_and_explicitly_merges(tmp_path):
    (tmp_path/"app.py").write_text("base")
    session=create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False))
    async def child_prompt(child,text,**kwargs):
        assert child.workspace_dir != session.workspace_dir
        assert not any(t.name in {"run_subagent","apply_subagent_changes"} for t in child.agent.state.tools)
        write=next(t for t in child.agent.state.tools if t.name == "write")
        await write.execute("w",{"path":"app.py","content":"worker"})
        return [AssistantMessage(content=[TextContent(text="implemented")])]
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with patch.object(AgentSession,"prompt",new=child_prompt):
        result=asyncio.run(dispatch.execute("dispatch",{"role":"worker","task":"implement"}))
    assert (tmp_path/"app.py").read_text() == "base"
    assert result.details["changed_files"] == ["app.py"]
    apply=next(t for t in session.agent.state.tools if t.name == "apply_subagent_changes")
    params={k:result.details[k] for k in ("lane_id","change_digest")}
    asyncio.run(apply.execute("apply",params))
    assert (tmp_path/"app.py").read_text() == "worker"
    with pytest.raises(ValueError,match="already applied"):
        asyncio.run(apply.execute("again",params))
    session.close()


def test_parent_read_only_cannot_delegate_writes(tmp_path):
    session=create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,read_only_mode=True))
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with pytest.raises(ValueError,match="read-only"):
        asyncio.run(dispatch.execute("d",{"role":"worker","task":"write"}))
    session.close()


def test_docker_cannot_read_host_secrets_or_publish_state(tmp_path):
    import os
    if os.environ.get("LOOPWEAVER_TEST_DOCKER") != "1":
        pytest.skip("Docker integration opt-in")
    (tmp_path/".env").write_text("top-secret")
    tools=sandbox_tools(create_builtin_tools(tmp_path,["bash"]),tmp_path,"loopweaver-sandbox:local",{})
    async def check():
        result=await tools[0].execute("b",{"command":"test ! -e .env && id -u && printf ok > app.txt && mkdir -p .loopweaver && printf bad > .loopweaver/state"})
        assert "65534" in result.content[0].text
    asyncio.run(check())
    assert (tmp_path/"app.txt").read_text() == "ok"
    assert not (tmp_path/".loopweaver"/"state").exists()
    assert (tmp_path/".env").read_text() == "top-secret"


def test_chain_reviewer_sees_worker_files(tmp_path):
    (tmp_path/"app.py").write_text("base")
    session=create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False))
    async def prompt(child,text,**kwargs):
        if text.startswith("implement"):
            write=next(t for t in child.agent.state.tools if t.name == "write")
            await write.execute("w",{"path":"app.py","content":"implemented"})
        else:
            assert (child.workspace_dir/"app.py").read_text() == "implemented"
            assert all(t.read_only for t in child.agent.state.tools)
        return [AssistantMessage(content=[TextContent(text="done")])]
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with patch.object(AgentSession,"prompt",new=prompt):
        result=asyncio.run(dispatch.execute("chain",{"mode":"chain","tasks":[{"role":"worker","task":"implement"},{"role":"reviewer","task":"review"}]}))
    assert len(result.details["children"]) == 2
    assert (tmp_path/"app.py").read_text() == "base"
    session.close()


def test_worker_survives_parent_restart_and_digest_detects_tampering(tmp_path):
    options=CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,session_id="parent")
    session=create_agent_session(options)
    lane=session._create_worker_lane("worker")
    lane_id=next(k for k,v in session._lanes.items() if v is lane)
    (lane.workspace_dir/"new.txt").write_text("first")
    original=session._worker_snapshots[lane_id].preview()
    session.store.append_journal_entry("worker_state",{"lane_id":lane_id,"status":"completed"})
    session.close_lane(lane_id)
    session.close()
    restored=create_agent_session(options)
    snapshot=restored._worker_snapshot(lane_id)
    (snapshot.target/"new.txt").write_text("second")
    apply=next(t for t in restored.agent.state.tools if t.name == "apply_subagent_changes")
    with pytest.raises(ValueError,match="digest"):
        asyncio.run(apply.execute("apply",{"lane_id":lane_id,"change_digest":original["change_digest"]}))
    inspect=next(t for t in restored.agent.state.tools if t.name == "inspect_subagent_changes")
    reviewed=asyncio.run(inspect.execute("inspect",{"lane_id":lane_id})).details
    assert "second" in reviewed["diff"]
    asyncio.run(apply.execute("apply",{"lane_id":lane_id,"change_digest":reviewed["change_digest"]}))
    assert (tmp_path/"new.txt").read_text() == "second"
    restored.close()


def test_docker_git_inspection_and_cancel_cleanup(tmp_path):
    import os
    if os.environ.get("LOOPWEAVER_TEST_DOCKER") != "1":
        pytest.skip("Docker integration opt-in")
    from agent_core.cancellation import CancellationToken
    subprocess.run(["git","init","-q",str(tmp_path)],check=True)
    (tmp_path/"app.txt").write_text("untracked")
    tools=sandbox_tools(create_builtin_tools(tmp_path,["git_status","bash"]),tmp_path,"loopweaver-sandbox:local",{})
    async def check():
        git=next(t for t in tools if t.name == "git_status")
        result=await git.execute("git",{})
        assert "app.txt" in result.content[0].text
        shell=next(t for t in tools if t.name == "bash")
        for mode in ("token","task"):
            token=CancellationToken()
            task=asyncio.create_task(shell.execute("sleep",{"command":"sleep 20; printf bad > should-not-publish"},token))
            await asyncio.sleep(0.3)
            token.cancel() if mode == "token" else task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    import coding_agent.sandbox as module
    original = module.docker_command
    names = []
    def capture(workspace,image,name,**kwargs):
        names.append(name)
        return original(workspace,image,name,**kwargs)
    with patch.object(module,"docker_command",side_effect=capture):
        asyncio.run(check())
    for name in names:
        assert subprocess.run(["docker","inspect",name],capture_output=True).returncode != 0
    assert not (tmp_path/"should-not-publish").exists()


def test_parallel_workers_are_independent_and_merge_disjoint_files(tmp_path):
    session=create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False))
    async def prompt(child,text,**kwargs):
        write=next(t for t in child.agent.state.tools if t.name == "write")
        await write.execute("w",{"path":text+".py","content":text})
        await asyncio.sleep(0.01)
        assert not (child.workspace_dir/("b.py" if text == "a" else "a.py")).exists()
        return [AssistantMessage(content=[TextContent(text="done")])]
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with patch.object(AgentSession,"prompt",new=prompt):
        result=asyncio.run(dispatch.execute("batch",{"mode":"parallel","tasks":[{"role":"worker","task":"a"},{"role":"worker","task":"b"}]}))
    assert not (tmp_path/"a.py").exists() and not (tmp_path/"b.py").exists()
    apply=next(t for t in session.agent.state.tools if t.name == "apply_subagent_changes")
    for details in result.details["children"]:
        asyncio.run(apply.execute("apply",{k:details[k] for k in ("lane_id","change_digest")}))
    assert (tmp_path/"a.py").read_text() == "a" and (tmp_path/"b.py").read_text() == "b"
    session.close()


def test_workspace_symlink_escape_is_rejected(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    outside=tmp_path/"outside"
    outside.write_text("secret")
    try:
        (source/"link").symlink_to(outside)
    except OSError:
        pytest.skip("OS does not grant symlink creation")
    with pytest.raises(ValueError,match="Unsafe"):
        WorkspaceSnapshot(source,tmp_path/"target")


def test_parallel_worker_approvals_are_namespaced_and_forwarded(tmp_path):
    from coding_agent.approval import ApprovalGate
    from agent_core import AgentContext, AgentLoopConfig
    from agent_core.agent_loop import _prepare_tool_call
    from ai.types import ToolCall
    gate=ApprovalGate()
    session=create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,approval_gate=gate))
    approvals=[]
    async def parent_listener(event):
        if event.get("type") == "approval_required":
            approvals.append(event)
            assert gate.set_requester(event["toolCallId"],"owner")
            assert gate.resolve(event["toolCallId"],True,actor_id="owner") == "approved"
    session.subscribe(parent_listener)
    async def prompt(child,text,**kwargs):
        tc=ToolCall(id="same-provider-id",name="write",arguments={"path":text+".txt","content":"ok"})
        assistant=AssistantMessage(content=[tc],stop_reason="toolUse")
        context=AgentContext(system_prompt="",messages=[],tools=child.agent.state.tools)
        config=AgentLoopConfig(model=child.agent.state.model,convert_to_llm=lambda m:m,approval_gate=child.approval_gate)
        prepared,_,_=await _prepare_tool_call(context,assistant,tc,config,child.agent._dispatch_event,None)
        assert prepared is not None
        return [AssistantMessage(content=[TextContent(text="approved")])]
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with patch.object(AgentSession,"prompt",new=prompt):
        asyncio.run(dispatch.execute("batch",{"mode":"parallel","tasks":[{"role":"worker","task":"a"},{"role":"worker","task":"b"}]}))
    assert len(approvals) == 2
    assert approvals[0]["toolCallId"] != approvals[1]["toolCallId"]
    assert all(e["delegated"] and e["sessionId"] == session.session_id for e in approvals)
    assert gate.pending() == []
    session.close()


def test_worker_inherits_docker_backend(tmp_path):
    import os
    if os.environ.get("LOOPWEAVER_TEST_DOCKER") != "1":
        pytest.skip("Docker integration opt-in")
    session=create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,tool_backend="docker"))
    async def prompt(child,text,**kwargs):
        assert child._options.tool_backend == "docker"
        write=next(t for t in child.agent.state.tools if t.name == "write")
        await write.execute("w",{"path":"sandboxed.py","content":"worker-in-docker"})
        return [AssistantMessage(content=[TextContent(text="done")])]
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with patch.object(AgentSession,"prompt",new=prompt):
        result=asyncio.run(dispatch.execute("dispatch",{"role":"worker","task":"implement"}))
    assert result.details["changed_files"] == ["sandboxed.py"]
    assert not (tmp_path/"sandboxed.py").exists()
    session.close()
