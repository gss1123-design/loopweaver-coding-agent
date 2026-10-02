"""Recoverable file publication: immutable backups + write-ahead file intents.

Not a multi-file atomic transaction. Recovery infers an interrupted replace
only when an intent exists and disk hashes match the frozen before/after pair.
External edits and damaged audit records fail closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .workspaces import WorkspaceSnapshot, files, protected


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Unsafe merge file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as fp:
        fp.write(data)
        fp.flush()
        os.fsync(fp.fileno())


@contextmanager
def file_lock(path: Path, *, create: bool = True):
    """OS-owned lock releases on process death; no stale PID-file guessing."""
    if create:
        path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b" if create else "r+b") as fp:
        if path.stat().st_size == 0:
            fp.write(b"0")
            fp.flush()
        fp.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fp.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(fp.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another workspace merge is running") from exc
        try:
            yield
        finally:
            fp.seek(0)
            if os.name == "nt":
                msvcrt.locking(fp.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(fp.fileno(),fcntl.LOCK_UN)


@contextmanager
def merge_lock(workspace: Path):
    runtime = workspace / ".xingclaw"
    if runtime.resolve() != runtime.absolute():
        raise ValueError("Unsafe runtime state directory")
    with file_lock(runtime / "merge.lock"):
        yield


def lease_held(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with file_lock(path,create=False):
            return False
    except (ValueError,OSError):
        return True  # An inaccessible lease is not permission to merge.


class MergeTransaction:
    def __init__(self, workspace: Path, root: Path, *, expected_digest: str | None = None):
        self.workspace = workspace.resolve(strict=True)
        self.root = root.absolute()
        if not self.root.resolve().is_relative_to(self.workspace / ".xingclaw" / "workspaces") or self.root.resolve() != self.root:
            raise ValueError("Unsafe merge transaction directory")
        self.plan = json.loads((self.root / "plan.json").read_text(encoding="utf-8"))
        if expected_digest is not None and digest(self.plan) != expected_digest:
            raise ValueError("Merge plan digest mismatch")
        if self.plan.get("schema_version") != 1 or not isinstance(self.plan.get("files"),list):
            raise ValueError("Invalid merge plan")
        seen = set()
        for item in self.plan["files"]:
            relative = item["path"]
            if not isinstance(relative,str) or not relative or relative != Path(relative).as_posix() or ":" in relative or "\\" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts or protected(relative) or relative in seen:
                raise ValueError("Unsafe merge plan path")
            seen.add(relative)
            self.destination(relative)

    @classmethod
    def prepare(cls, snapshot: WorkspaceSnapshot, root: Path) -> "MergeTransaction":
        if root.resolve() != root.absolute() or not root.resolve().is_relative_to(snapshot.source / ".xingclaw" / "workspaces"):
            raise ValueError("Unsafe merge transaction directory")
        preview = snapshot.preview()
        if preview["conflicts"]:
            raise ValueError("Workspace merge conflict: " + ", ".join(preview["conflicts"]))
        # Reuse publication path validation before creating a plan; no writes
        # to the project occur here. Plan lives outside the writable candidate.
        if root.exists():
            raise ValueError("Merge transaction already exists")
        root.mkdir(parents=True)
        candidate = files(snapshot.target)
        plan = {"schema_version":1,"change_digest":preview["change_digest"],"files":[]}
        for index,relative in enumerate(preview["changed_files"]):
            old,new = snapshot.source / relative,snapshot.target / relative
            before,after = snapshot.base.get(relative),candidate.get(relative)
            mode = old.stat().st_mode & 0o777 if old.exists() else 0o644 | (new.stat().st_mode & 0o111)
            for kind,source,expected in (("before",old,before),("after",new,after)):
                if expected is not None:
                    backup = root / f"{index}.{kind}"
                    durable_write(backup,source.read_bytes())
                    if file_hash(backup) != expected:
                        raise ValueError("Files changed while preparing merge")
            plan["files"].append({"path":relative,"before":before,"after":after,"mode":mode})
        durable_write(root / "plan.json",json.dumps(plan,ensure_ascii=False).encode())
        transaction = cls(snapshot.source,root)
        transaction.append("prepared")
        return transaction

    def destination(self, relative: str) -> Path:
        target = self.workspace / relative
        if not target.resolve().is_relative_to(self.workspace) or target.is_symlink():
            raise ValueError("Merge path escapes workspace")
        if target.exists() and not target.is_file():
            raise ValueError("Merge would replace a directory")
        for parent in target.parents:
            if parent == self.workspace:
                break
            if parent.is_symlink() or (hasattr(parent,"is_junction") and parent.is_junction()) or (parent.exists() and not parent.is_dir()):
                raise ValueError("Unsafe merge parent directory")
        return target

    def events(self) -> list[dict]:
        path = self.root / "events.jsonl"
        if not path.exists():
            return []
        result = []
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break  # Only an unterminated tail is recoverable corruption.
            try:
                value = json.loads(line)
            except (ValueError,UnicodeDecodeError) as exc:
                raise ValueError("Corrupt merge journal; manual inspection required") from exc
            if not isinstance(value,dict) or value.get("seq") != len(result)+1 or value.get("type") not in {"prepared","apply_started","rollback_started","file_intent","file_done","committed","rolled_back"}:
                raise ValueError("Invalid merge journal sequence/event")
            if value["type"] in {"file_intent","file_done"}:
                if value.get("path") not in {i["path"] for i in self.plan["files"]} or value.get("direction") not in {"apply","rollback"}:
                    raise ValueError("Invalid merge file event")
                if value["type"] == "file_done" and not any(e["type"] == "file_intent" and e.get("path") == value["path"] and e.get("direction") == value["direction"] for e in result):
                    raise ValueError("Merge completion has no intent")
            if value["type"] == "committed" and not all(any(e["type"] == "file_done" and e.get("direction") == "apply" and e.get("path") == item["path"] for e in result) for item in self.plan["files"]):
                raise ValueError("Merge commit has incomplete files")
            result.append(value)
        return result

    def append(self, event: str, **payload):
        events = self.events()
        path = self.root / "events.jsonl"
        data = path.read_bytes() if path.exists() else b""
        boundary = data.rfind(b"\n") + 1
        # Remove only the uncommitted tail before appending a new complete line.
        with path.open("r+b" if path.exists() else "w+b") as fp:
            fp.truncate(boundary)
            fp.seek(boundary)
            fp.write((json.dumps({"seq":len(events)+1,"type":event,**payload})+"\n").encode())
            fp.flush()
            os.fsync(fp.fileno())

    def status(self) -> dict:
        events = self.events()
        kinds = [e["type"] for e in events]
        state = "rolled_back" if "rolled_back" in kinds else "committed" if "committed" in kinds and "rollback_started" not in kinds else "rolling_back" if "rollback_started" in kinds else "applying" if "apply_started" in kinds else "prepared"
        return {"state":state,"files":[i["path"] for i in self.plan["files"]],
                "applied":[e["path"] for e in events if e["type"] == "file_done" and e["direction"] == "apply"],
                "restored":[e["path"] for e in events if e["type"] == "file_done" and e["direction"] == "rollback"],
                "plan_digest":digest(self.plan),"change_digest":self.plan["change_digest"]}

    def _replace(self, index: int, item: dict, direction: str):
        target = self.destination(item["path"])
        kind = "after" if direction == "apply" else "before"
        expected = item[kind]
        if expected is None:
            if target.exists():
                target.unlink()
            return
        backup = self.root / f"{index}.{kind}"
        if file_hash(backup) != expected:
            raise ValueError("Merge backup checksum mismatch")
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary = tempfile.mkstemp(prefix=".xingclaw-merge-",dir=target.parent)
        try:
            with os.fdopen(fd,"wb") as fp:
                fp.write(backup.read_bytes())
                fp.flush()
                os.fsync(fp.fileno())
            Path(temporary).chmod(item["mode"])
            os.replace(temporary,target)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()

    def recover(self, action: str = "resume") -> dict:
        if action not in {"resume","rollback"}:
            raise ValueError("Unknown merge recovery action")
        with merge_lock(self.workspace):
            events = self.events()
            state = self.status()["state"]
            if (action == "resume" and state == "committed") or (action == "rollback" and state == "rolled_back"):
                return self.status()
            if action == "resume" and state in {"rolling_back","rolled_back"}:
                raise ValueError("Rollback already started; cannot resume application")
            direction = "apply" if action == "resume" else "rollback"
            intended = {e["path"] for e in events if e["type"] == "file_intent" and e["direction"] == direction}
            done = {e["path"] for e in events if e["type"] == "file_done" and e["direction"] == direction}
            touched = {e["path"] for e in events if e["type"] == "file_intent" and e["direction"] == "apply"}
            selected = [(i,item) for i,item in enumerate(self.plan["files"]) if direction == "apply" or item["path"] in touched]
            # Preflight the entire set before any more writes; preserve external edits.
            for index,item in selected:
                current = file_hash(self.destination(item["path"]))
                target_hash = item["after" if direction == "apply" else "before"]
                allowed = {target_hash} if item["path"] in done else {item["before"],item["after"]} if item["path"] in intended or direction == "rollback" else {item["before"]}
                if current not in allowed:
                    raise ValueError("Merge recovery conflict: " + item["path"])
                for kind in ("before","after"):
                    if item[kind] is not None and file_hash(self.root / f"{index}.{kind}") != item[kind]:
                        raise ValueError("Merge backup checksum mismatch")
            self.append("apply_started" if direction == "apply" else "rollback_started")
            for index,item in selected:
                if item["path"] in done:
                    continue
                self.append("file_intent",path=item["path"],direction=direction)
                desired = item["after" if direction == "apply" else "before"]
                current = file_hash(self.destination(item["path"]))
                if current not in {item["before"],item["after"]}:
                    raise ValueError("File changed during merge: " + item["path"])
                if current != desired:
                    self._replace(index,item,direction)
                self.append("file_done",path=item["path"],direction=direction)
            for _,item in selected:
                if file_hash(self.destination(item["path"])) != item["after" if direction == "apply" else "before"]:
                    raise ValueError("File changed before merge completion: " + item["path"])
            self.append("committed" if direction == "apply" else "rolled_back")
            return self.status()
