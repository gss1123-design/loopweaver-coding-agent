"""Fixed crash-boundary contracts: no live model or external side effects."""
import asyncio
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
from ai.models import get_model
from ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from coding_agent.factory import create_agent_session
from coding_agent.types import CreateAgentSessionOptions
from coding_agent.recovery import batch_key
from coding_agent.serde import message_to_dict


def open_session(tmp_path):
    return create_agent_session(CreateAgentSessionOptions(workspace_dir=tmp_path,
        session_id="recovery_test",model=get_model("openai-standard","gpt-4o-mini"),load_workspace_resources=False))


def commit(session,message):
    asyncio.run(session._on_agent_event({"type":"message_end","message":message,"turnId":2}))


@pytest.mark.parametrize("final",[False,True])
def test_restart_recovers_unprojected_message_without_replaying_tools(tmp_path,final):
    session=open_session(tmp_path)
    commit(session,UserMessage(content="task"))
    assistant=AssistantMessage(content=[TextContent(text="finished")] if final else [ToolCall(id="w",name="write")],stop_reason="stop" if final else "toolUse")
    with patch.object(session.store,"append_context_message",side_effect=OSError("crash before projection")):
        with pytest.raises(OSError):
            commit(session,assistant)
    session.close()
    restored=open_session(tmp_path)
    status=restored.recovery_status()
    assert status["pending_projection"]
    assert status["recoverable"] == final
    with patch.object(restored,"continue_run",new=AsyncMock()) as continued,patch.object(restored,"prompt",new=AsyncMock()) as prompt:
        if final:
            assert asyncio.run(restored.resume_run()) == []
        else:
            with pytest.raises(ValueError,match="副作用"):
                asyncio.run(restored.resume_run())
        continued.assert_not_awaited()
        prompt.assert_not_awaited()
    assert isinstance(restored.messages[-1],AssistantMessage)
    restored.close()


def test_committed_result_before_projection_is_reused_after_restart(tmp_path):
    session=open_session(tmp_path)
    commit(session,UserMessage(content="task"))
    commit(session,AssistantMessage(content=[ToolCall(id="w",name="write")],stop_reason="toolUse"))
    with patch.object(session.store,"append_context_message",side_effect=OSError("crash")):
        with pytest.raises(OSError):
            commit(session,ToolResultMessage(tool_call_id="w",tool_name="write",content=[TextContent(text="ok")]))
    session.close()
    restored=open_session(tmp_path)
    before=restored.store.load_journal()
    assert restored.recovery_status()["recoverable"]
    assert restored.store.load_journal() == before  # inspection has no writes
    with patch.object(restored,"continue_run",new=AsyncMock(return_value=[])) as continued:
        asyncio.run(restored.resume_run())
        continued.assert_awaited_once()
    assert len([m for m in restored.messages if isinstance(m,ToolResultMessage)]) == 1
    restored.close()


@pytest.mark.parametrize("started",[False,True])
def test_missing_result_never_auto_replays_even_readonly_tools(tmp_path,started):
    session=open_session(tmp_path)
    assistant=AssistantMessage(content=[ToolCall(id="r",name="read")],stop_reason="toolUse")
    commit(session,UserMessage(content="task"))
    commit(session,assistant)
    if started:
        asyncio.run(session._on_agent_event({"type":"tool_checkpoint_start","toolCallId":"r","toolName":"read","effectiveArgs":{}}))
    session.agent.set_messages(session.store.load_session_messages())
    status=session.recovery_status()
    assert not status["recoverable"]
    assert status["tools"][0]["state"] == ("uncertain" if started else "not_confirmed_started")
    with patch.object(session,"continue_run",new=AsyncMock()) as continued:
        with pytest.raises(ValueError):
            asyncio.run(session.resume_run())
        continued.assert_not_awaited()
    session.close()


@pytest.mark.parametrize("damage",[b'{"seq":999',b'corrupt\n'])
def test_damaged_journal_blocks_model_request(tmp_path,damage):
    session=open_session(tmp_path)
    commit(session,UserMessage(content="task"))
    session.agent.set_messages(session.store.load_session_messages())
    with session.store.journal_file.open("ab") as fp:
        fp.write(damage)
    assert not session.recovery_status()["recoverable"]
    with patch.object(session,"continue_run",new=AsyncMock()) as continued:
        with pytest.raises(ValueError,match="日志"):
            asyncio.run(session.resume_run())
        continued.assert_not_awaited()
    session.close()


def test_branch_switch_does_not_project_checkpoint_from_other_leaf(tmp_path):
    session=open_session(tmp_path)
    commit(session,UserMessage(content="first"))
    first=session.store.get_leaf_id()
    commit(session,UserMessage(content="second"))
    with patch.object(session.store,"append_context_message",side_effect=OSError("crash")):
        with pytest.raises(OSError):
            commit(session,AssistantMessage(content=[TextContent(text="other branch")]))
    session.switch_to_entry(first)
    assert not session.recovery_status()["pending_projection"]
    session.close()


def test_conflicting_result_records_block_recovery(tmp_path):
    session=open_session(tmp_path)
    assistant=AssistantMessage(content=[ToolCall(id="a",name="write")],stop_reason="toolUse")
    commit(session,assistant)
    commit(session,ToolResultMessage(tool_call_id="a",tool_name="write",content=[TextContent(text="first")]))
    session.store.append_journal_entry("tool_result_committed",{"batch_key":batch_key(assistant),"message":message_to_dict(ToolResultMessage(tool_call_id="a",tool_name="write",content=[TextContent(text="different")]))})
    session.agent.set_messages(session.store.load_session_messages())
    assert not session.recovery_status()["recoverable"]
    with pytest.raises(ValueError,match="冲突"):
        asyncio.run(session.resume_run())
    session.close()


def test_cli_query_and_running_resume_guard(tmp_path):
    from coding_agent.runner import _handle_interactive_command
    session=open_session(tmp_path)
    commit(session,UserMessage(content="task"))
    session.agent.set_messages(session.store.load_session_messages())
    output=[]
    asyncio.run(_handle_interactive_command(session,"/recovery",output=output.append))
    assert '"recoverable": true' in output[-1]
    session.agent.state.is_streaming=True
    try:
        assert not session.recovery_status()["recoverable"]
        with pytest.raises(ValueError,match="仍在运行"):
            asyncio.run(session.resume_run())
    finally:
        session.agent.state.is_streaming=False
        session.close()
