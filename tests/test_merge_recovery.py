import asyncio
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from coding_agent.workspaces import WorkspaceSnapshot
from coding_agent.merge_transaction import MergeTransaction,digest,merge_lock
from coding_agent.factory import create_agent_session
from coding_agent.types import CreateAgentSessionOptions
from coding_agent.agent_session import AgentSession
from ai.models import get_model
from ai.types import AssistantMessage,TextContent


class ProcessCrash(BaseException):
    pass


def setup_merge(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    (source/"a").write_text("old-a")
    (source/"b").write_text("old-b")
    snapshot=WorkspaceSnapshot(source,tmp_path/"candidate")
    (snapshot.target/"a").write_text("new-a")
    (snapshot.target/"b").write_text("new-b")
    root=source/".xingclaw"/"workspaces"/"test"/"merges"/"test"
    return snapshot,MergeTransaction.prepare(snapshot,root)


def crash_after_replace(transaction):
    append=transaction.append
    def interrupted(event,**payload):
        if event == "file_done" and payload["path"] == "a":
            raise ProcessCrash()
        append(event,**payload)
    with patch.object(transaction,"append",side_effect=interrupted):
        with pytest.raises(ProcessCrash):
            transaction.recover()


def test_replace_before_completion_record_recovers_without_replacing_again(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    crash_after_replace(transaction)
    assert (snapshot.source/"a").read_text() == "new-a"
    assert (snapshot.source/"b").read_text() == "old-b"
    (snapshot.target/"b").write_text("candidate-was-modified")
    restored=MergeTransaction(snapshot.source,transaction.root,expected_digest=digest(transaction.plan))
    replace=restored._replace
    calls=[]
    def count(index,item,direction):
        calls.append(item["path"])
        replace(index,item,direction)
    with patch.object(restored,"_replace",side_effect=count):
        assert restored.recover()["state"] == "committed"
        assert restored.recover()["state"] == "committed"
    assert calls == ["b"]
    assert (snapshot.source/"b").read_text() == "new-b"


def test_partial_merge_can_roll_back_from_backups(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    crash_after_replace(transaction)
    assert MergeTransaction(snapshot.source,transaction.root).recover("rollback")["state"] == "rolled_back"
    assert (snapshot.source/"a").read_text() == "old-a"
    assert (snapshot.source/"b").read_text() == "old-b"
    with pytest.raises(ValueError,match="Rollback"):
        transaction.recover("resume")


def test_conflict_prevents_resume_and_rollback_overwriting_user_edits(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    crash_after_replace(transaction)
    (snapshot.source/"a").write_text("user-edit")
    for action in ("resume","rollback"):
        with pytest.raises(ValueError,match="conflict"):
            transaction.recover(action)
    assert (snapshot.source/"a").read_text() == "user-edit"
    assert (snapshot.source/"b").read_text() == "old-b"


def test_intent_fsync_failure_runs_no_replace(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    append=transaction.append
    def disk_full(event,**payload):
        if event == "file_intent":
            raise OSError("disk full")
        append(event,**payload)
    with patch.object(transaction,"append",side_effect=disk_full),patch.object(transaction,"_replace") as replace:
        with pytest.raises(OSError,match="disk full"):
            transaction.recover()
        replace.assert_not_called()
    assert (snapshot.source/"a").read_text() == "old-a"


def test_backup_damage_blocks_all_writes(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    (transaction.root/"1.after").write_text("damaged")
    with pytest.raises(ValueError,match="checksum"):
        transaction.recover()
    assert (snapshot.source/"a").read_text() == "old-a"


def test_truncated_tail_is_repaired_but_corrupt_middle_fails_closed(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    crash_after_replace(transaction)
    journal=transaction.root/"events.jsonl"
    with journal.open("ab") as fp:
        fp.write(b'{"seq":99')
    assert transaction.recover()["state"] == "committed"
    with journal.open("ab") as fp:
        fp.write(b'not-json\n')
    with pytest.raises(ValueError,match="Corrupt"):
        transaction.recover("rollback")


def test_workspace_merge_lock_is_shared_and_released(tmp_path):
    with merge_lock(tmp_path):
        with pytest.raises(ValueError,match="Another"):
            with merge_lock(tmp_path):
                pass
    with merge_lock(tmp_path):
        pass


def test_create_delete_and_rollback_committed_merge(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    (source/"deleted").write_text("keep-backup")
    snapshot=WorkspaceSnapshot(source,tmp_path/"candidate")
    (snapshot.target/"deleted").unlink()
    (snapshot.target/"added").write_text("new")
    transaction=MergeTransaction.prepare(snapshot,source/".xingclaw"/"workspaces"/"t"/"merge")
    transaction.recover()
    assert not (source/"deleted").exists() and (source/"added").exists()
    transaction.recover("rollback")
    assert (source/"deleted").read_text() == "keep-backup" and not (source/"added").exists()


def test_worker_state_and_pending_merge_recover_after_parent_restart(tmp_path):
    options=CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,session_id="parent")
    (tmp_path/"a").write_text("old")
    session=create_agent_session(options)
    async def child_prompt(child,text,**kwargs):
        (child.workspace_dir/"a").write_text("new")
        return [AssistantMessage(content=[TextContent(text="saved-worker-result")])]
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with patch.object(AgentSession,"prompt",new=child_prompt):
        result=asyncio.run(dispatch.execute("worker",{"role":"worker","task":"implement"}))
    lane_id=result.details["lane_id"]
    apply=next(t for t in session.agent.state.tools if t.name == "apply_subagent_changes")
    original=MergeTransaction.append
    def interrupted(self,event,**payload):
        if event == "file_done":
            raise ProcessCrash()
        original(self,event,**payload)
    with patch.object(MergeTransaction,"append",new=interrupted):
        with pytest.raises(ProcessCrash):
            asyncio.run(apply.execute("merge",{k:result.details[k] for k in ("lane_id","change_digest")}))
    session.close()
    restored=create_agent_session(options)
    worker=restored.list_workers()[0]
    assert worker["status"] == "completed" and "saved-worker-result" in worker["result"]
    assert worker["merge"]["state"] == "applying"
    apply=next(t for t in restored.agent.state.tools if t.name == "apply_subagent_changes")
    with pytest.raises(ValueError,match="Unfinished"):
        asyncio.run(apply.execute("retry",{k:result.details[k] for k in ("lane_id","change_digest")}))
    recover=next(t for t in restored.agent.state.tools if t.name == "recover_subagent_merge")
    params={"lane_id":lane_id,"plan_digest":worker["merge"]["plan_digest"],"action":"resume"}
    assert asyncio.run(recover.execute("recover",params)).details["state"] == "committed"
    assert restored.list_workers()[0]["status"] == "applied"
    params["action"]="rollback"
    asyncio.run(recover.execute("rollback",params))
    assert (tmp_path/"a").read_text() == "old"
    restored.close()


def test_cli_worker_queries_are_read_only_and_validate_usage(tmp_path):
    import json
    from coding_agent.runner import _handle_interactive_command
    session=create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False))
    child=session._create_worker_lane("query")
    lane_id=next(k for k,v in session._lanes.items() if v is child)
    session.close_lane(lane_id)
    before=session.store.load_journal()
    output=[]
    assert asyncio.run(_handle_interactive_command(session,"/workers",output=output.append)) == (True,None)
    assert json.loads(output[-1])[0]["status"] == "interrupted"
    asyncio.run(_handle_interactive_command(session,f"/worker {lane_id}",output=output.append))
    assert json.loads(output[-1])["worker"]["lane_id"] == lane_id
    asyncio.run(_handle_interactive_command(session,"/worker",output=output.append))
    assert "usage:" in output[-1]
    assert session.store.load_journal() == before
    session.close()


def test_other_client_sees_worker_lease_and_restart_marks_interruption(tmp_path):
    options=CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,session_id="parent")
    parent=create_agent_session(options)
    parent._create_worker_lane("unfinished")
    other=create_agent_session(options)
    assert other.list_workers()[0]["status"] == "running"
    assert other.list_workers()[0]["active"]
    parent.close()
    assert other.list_workers()[0]["status"] == "interrupted"
    assert not other.list_workers()[0]["active"]
    other.close()


def test_rollback_itself_can_resume_after_replace_before_record(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    transaction.recover()
    append=transaction.append
    def interrupted(event,**payload):
        if event == "file_done" and payload.get("direction") == "rollback":
            raise ProcessCrash()
        append(event,**payload)
    with patch.object(transaction,"append",side_effect=interrupted):
        with pytest.raises(ProcessCrash):
            transaction.recover("rollback")
    assert transaction.status()["state"] == "rolling_back"
    assert MergeTransaction(snapshot.source,transaction.root).recover("rollback")["state"] == "rolled_back"
    assert (snapshot.source/"a").read_text() == "old-a"
    assert (snapshot.source/"b").read_text() == "old-b"


def test_partial_rollback_does_not_touch_unattempted_user_change(tmp_path):
    snapshot,transaction=setup_merge(tmp_path)
    crash_after_replace(transaction)
    (snapshot.source/"b").write_text("user-edit")
    transaction.recover("rollback")
    assert (snapshot.source/"b").read_text() == "user-edit"


@pytest.mark.parametrize("outcome",["failed","cancelled","timeout"])
def test_worker_terminal_failure_is_persisted_and_not_normally_mergeable(tmp_path,outcome):
    options=CreateAgentSessionOptions(workspace_dir=tmp_path,model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False,session_id="parent")
    session=create_agent_session(options)
    async def fail(child,text,**kwargs):
        (child.workspace_dir/"candidate").write_text("unfinished")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        if outcome == "timeout":
            await asyncio.sleep(1)
        raise RuntimeError("test failure")
    session.subagent_timeout_seconds=0.01 if outcome == "timeout" else 0
    dispatch=next(t for t in session.agent.state.tools if t.name == "run_subagent")
    with patch.object(AgentSession,"prompt",new=fail):
        with pytest.raises(asyncio.CancelledError if outcome == "cancelled" else RuntimeError):
            asyncio.run(dispatch.execute("worker",{"role":"worker","task":"implement"}))
    session.close()
    restored=create_agent_session(options)
    worker=restored.list_workers()[0]
    assert worker["status"] == ("failed" if outcome == "timeout" else outcome)
    assert not worker["active"]
    preview=restored.inspect_worker(worker["lane_id"])
    apply=next(t for t in restored.agent.state.tools if t.name == "apply_subagent_changes")
    with pytest.raises(ValueError,match="did not complete"):
        asyncio.run(apply.execute("merge",{"lane_id":worker["lane_id"],"change_digest":preview["change_digest"]}))
    assert not (tmp_path/"candidate").exists()
    restored.close()
