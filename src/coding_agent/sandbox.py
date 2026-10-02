"""Opt-in Docker backend for builtin tools; no silent host fallback."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import uuid
import tempfile

from agent_core import AgentTool, AgentToolResult
from agent_core.cancellation import await_with_cancellation, throw_if_cancelled
from ai.types import ToolResultMessage
from .serde import message_from_dict
from .workspaces import WorkspaceSnapshot, copy_git_metadata


def docker_command(workspace: Path, image: str, name: str, *, read_only: bool, runtime_source: Path | None = None) -> list[str]:
    workspace = workspace.resolve(strict=True)
    source = runtime_source or Path(__file__).resolve().parents[1]
    if not image or image.startswith("-") or "," in str(workspace):
        raise ValueError("Invalid sandbox image or workspace")
    mount = f"type=bind,source={workspace},target=/workspace" + (",readonly" if read_only else "")
    return ["docker", "create", "--rm", "-i", "--name", name,
            "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "512m", "--cpus", "1", "--pids-limit", "128", "--read-only",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,nosuid,noexec,size=64m", "--mount", mount,
            "--mount", f"type=bind,source={source},target=/opt/loopweaver,readonly",
            "--workdir", "/workspace", "--env", "PYTHONPATH=/opt/loopweaver",
            "--env", "GIT_OPTIONAL_LOCKS=0", "--env", "GIT_CONFIG_COUNT=1",
            "--env", "GIT_CONFIG_KEY_0=safe.directory", "--env", "GIT_CONFIG_VALUE_0=/workspace",
            image, "python", "-m", "coding_agent.sandbox_worker"]


def sandbox_tools(tools: list[AgentTool], workspace: Path, image: str, policy: dict) -> list[AgentTool]:
    def wrap(tool):
        async def execute(call_id, params, signal=None, on_update=None):
            # Existing host path hooks reject escapes. Translate workspace-local
            # absolute paths so container paths do not accidentally reference host.
            args = dict(params)
            for key in ("path", "cwd"):
                value = args.get(key)
                if isinstance(value, str) and Path(value).is_absolute():
                    args[key] = str(Path(value).resolve().relative_to(workspace.resolve())).replace("\\", "/")
            name = "loopweaver-" + uuid.uuid4().hex
            temporary = tempfile.TemporaryDirectory(prefix="loopweaver-sandbox-")
            try:
                snapshot = WorkspaceSnapshot(workspace, Path(temporary.name) / "workspace")
                runtime = WorkspaceSnapshot(Path(__file__).resolve().parents[1],Path(temporary.name)/"runtime")
                if tool.name in {"git_status", "git_diff"}:
                    copy_git_metadata(workspace, snapshot.target)
                # Docker on Linux must allow the unprivileged guest user to
                # access the dedicated scratch copy, never chmod host files.
                snapshot.target.chmod(0o777)
                for path in snapshot.target.rglob("*"):
                    path.chmod(0o777 if path.is_dir() else 0o666 | (path.stat().st_mode & 0o111))
                command = docker_command(snapshot.target, image, name, read_only=tool.read_only, runtime_source=runtime.target)
            except BaseException:
                temporary.cleanup()
                raise
            proc = None
            creator = None
            creator_spawn = None
            creation = None
            communication = None
            try:
                # Finish creation before cancellation/start. A canceled run
                # CLI can otherwise leave a late-registered 'created' container.
                creator_spawn = asyncio.create_task(asyncio.create_subprocess_exec(*command,
                        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE))
                creator = await asyncio.shield(creator_spawn)
                creation = asyncio.create_task(creator.communicate())
                _, create_error = await asyncio.wait_for(asyncio.shield(creation),timeout=30)
                if creator.returncode:
                    raise RuntimeError("Docker container creation failed: " + create_error.decode(errors="replace")[-2000:])
                throw_if_cancelled(signal)
                proc = await asyncio.create_subprocess_exec("docker","start","--attach","--interactive",name,
                        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                request = json.dumps({"name": tool.name, "args": args, "call_id": call_id, "policy": policy}).encode()
                communication = asyncio.create_task(proc.communicate(request))
                out, err = await await_with_cancellation(asyncio.wait_for(communication, timeout=120), signal)
                if proc.returncode:
                    raise RuntimeError("Docker sandbox failed: " + err.decode(errors="replace")[-2000:])
                result = message_from_dict(json.loads(out))
                if not isinstance(result, ToolResultMessage):
                    raise RuntimeError("Invalid sandbox response")
                if result.is_error:
                    raise RuntimeError("; ".join(b.text for b in result.content if hasattr(b, "text")))
                changed = snapshot.changes()
                if tool.read_only and changed:
                    raise RuntimeError("Read-only sandbox modified workspace")
                if changed:
                    snapshot.publish()
                return AgentToolResult(content=result.content, details=result.details)
            finally:
                # Explicitly remove the uniquely named container even if CLI
                # cancellation occurred after container creation.
                try:
                    if creator is None and creator_spawn is not None:
                        creator = await asyncio.wait_for(asyncio.shield(creator_spawn),timeout=10)
                    if creator is not None and creation is None:
                        creation = asyncio.create_task(creator.communicate())
                    if creation is not None and not creation.done():
                        await asyncio.wait_for(asyncio.shield(creation),timeout=30)
                    if communication is not None:
                        if not communication.done():
                            communication.cancel()
                        await asyncio.gather(communication,return_exceptions=True)
                    cleanup = await asyncio.create_subprocess_exec("docker", "rm", "-f", name,
                            stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
                    try:
                        await asyncio.wait_for(cleanup.wait(),timeout=10)
                    except asyncio.TimeoutError:
                        cleanup.kill()
                        await cleanup.wait()
                        raise RuntimeError(f"Container cleanup timed out; inspect {name}")
                finally:
                    if proc is not None and proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    if creator is not None and creator.returncode is None:
                        creator.kill()
                        await creator.wait()
                    if creation is not None and not creation.done():
                        creation.cancel()
                        await asyncio.gather(creation,return_exceptions=True)
                    temporary.cleanup()
        wrapped = replace(tool, execute=execute)
        setattr(wrapped, "_loopweaver_builtin_tool", True)
        return wrapped
    return [wrap(tool) for tool in tools]
