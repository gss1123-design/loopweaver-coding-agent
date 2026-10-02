"""Bounded workspace snapshots and optimistic, conflict-checked publication.

This is filesystem isolation, not an OS sandbox. Shell confinement is provided
separately by the Docker backend. Never publish runtime state or credentials.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import json
import difflib
import subprocess

EXCLUDED = {".git", ".xingclaw", ".ssh", ".aws", ".codex", ".venv", "venv",
            "node_modules", "__pycache__", ".pytest_cache", ".eval"}


def protected(relative: str) -> bool:
    parts = Path(relative).parts
    return any(p.casefold() in EXCLUDED or p.casefold().startswith((".env", "credentials", "secrets"))
               or p.casefold().endswith((".pem", ".key")) for p in parts)


def files(root: Path, *, max_bytes: int = 128 * 1024 * 1024) -> dict[str, str]:
    root = root.resolve(strict=True)
    found, size = {}, 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if protected(rel):
                dirs.remove(name)
                continue
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or not path.resolve().is_relative_to(root):
                raise ValueError(f"Workspace links are not supported: {rel}")
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if protected(rel):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError(f"Unsafe workspace file: {rel}")
            size += path.stat().st_size
            if size > max_bytes or len(found) >= 10000:
                raise ValueError("Workspace snapshot exceeds 128 MiB or 10000 files; use a smaller task workspace")
            found[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


class WorkspaceSnapshot:
    def __init__(self, source: Path, target: Path):
        self.source = source.resolve(strict=True)
        self.target = target.resolve()
        if self.target.exists():
            raise ValueError("Snapshot target already exists")
        self.base = files(self.source)
        self.target.mkdir(parents=True)
        for relative, digest in self.base.items():
            src = self.source / relative
            dst = self.target / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Workspace changed while creating snapshot; retry")

    def changes(self) -> list[str]:
        current = files(self.target)
        return sorted(p for p in self.base.keys() | current.keys() if self.base.get(p) != current.get(p))

    def preview(self) -> dict:
        candidate = files(self.target)
        host = files(self.source)
        changes = sorted(p for p in self.base.keys() | candidate.keys() if self.base.get(p) != candidate.get(p))
        manifest = {p:candidate.get(p) for p in changes}
        digest = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        diff = []
        budget = 8000
        for path in changes:
            old = self.source / path
            new = self.target / path
            if (old.exists() and old.stat().st_size > 32000) or (new.exists() and new.stat().st_size > 32000):
                chunk = f"{path}: large file; inspect separately\n"
            else:
                try:
                    left = old.read_text(encoding="utf-8").splitlines(keepends=True) if old.exists() else []
                    right = new.read_text(encoding="utf-8").splitlines(keepends=True) if new.exists() else []
                    chunk = "".join(difflib.unified_diff(left,right,fromfile="current-parent/"+path,tofile="worker/"+path))
                except UnicodeDecodeError:
                    chunk = f"{path}: binary change\n"
            diff.append(chunk[:budget])
            budget -= len(diff[-1])
            if budget <= 0:
                break
        return {"changed_files":changes,"change_digest":digest,
                "conflicts":[p for p in changes if host.get(p) != self.base.get(p)],
                "diff":"".join(diff),"diff_truncated":budget <= 0}

    def publish(self) -> list[str]:
        candidate = files(self.target)
        changes = sorted(p for p in self.base.keys() | candidate.keys() if self.base.get(p) != candidate.get(p))
        host = files(self.source)
        conflicts = [p for p in changes if host.get(p) != self.base.get(p)]
        if conflicts:
            raise ValueError("Workspace merge conflict: " + ", ".join(conflicts))
        # Validate every destination before any write. This also rejects a
        # parent file replacing a directory needed by another candidate file.
        for relative in changes:
            dst = self.source / relative
            if protected(relative) or not dst.resolve().is_relative_to(self.source):
                raise ValueError("Unsafe merge destination")
            if dst.exists() and dst.is_dir():
                raise ValueError(f"Merge would replace a directory: {relative}")
            for parent in dst.parents:
                if parent == self.source:
                    break
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"Merge parent is not a directory: {relative}")
        # Each file replace is atomic, but the batch is not a filesystem
        # transaction. The base records hashes, not an automatic rollback.
        for relative in changes:
            dst = self.source / relative
            if relative not in candidate:
                dst.unlink()
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".xingclaw-merge-", dir=dst.parent)
                os.close(fd)
                try:
                    shutil.copy2(self.target / relative, temporary)
                    mode = dst.stat().st_mode & 0o777 if dst.exists() else 0o644 | (Path(temporary).stat().st_mode & 0o111)
                    Path(temporary).chmod(mode)
                    os.replace(temporary, dst)
                finally:
                    if Path(temporary).exists():
                        Path(temporary).unlink()
        return changes


def copy_git_metadata(source: Path, target: Path) -> None:
    """Copy inspection metadata only; never mount host Git hooks or config."""
    git = source / ".git"
    if not git.exists():
        return
    if not git.is_dir() or git.is_symlink() or git.resolve() != git.absolute():
        raise ValueError("Git inspection currently requires a normal checkout, not a linked worktree")
    total = 0
    for directory, dirs, names in os.walk(git, followlinks=False):
        (target / ".git" / Path(directory).relative_to(git)).mkdir(parents=True, exist_ok=True)
        dirs[:] = [d for d in dirs if d not in {"hooks", "logs"}]
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink() or not path.resolve().is_relative_to(git.resolve()):
                raise ValueError("Unsafe Git metadata link")
        for name in names:
            if name in {"config", "config.worktree", "commondir", "gitdir"}:
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.resolve().is_relative_to(git.resolve()):
                raise ValueError("Unsafe Git metadata file")
            total += path.stat().st_size
            if total > 128 * 1024 * 1024:
                raise ValueError("Git inspection metadata exceeds 128 MiB")
            dst = target / ".git" / path.relative_to(git)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def initialize_snapshot_git(target: Path) -> None:
    """Create a local baseline without parent history, hooks or Git credentials."""
    env = {**os.environ,"GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":os.devnull}
    prefix = ["git","-c","core.autocrlf=false","-c","core.hooksPath="+os.devnull,
              "-c","commit.gpgsign=false","-c","user.name=XingClaw Worker",
              "-c","user.email=worker@localhost"]
    for args in (["init","--quiet","--template="],["add","--all"],["commit","--quiet","--allow-empty","-m","Worker baseline"]):
        result = subprocess.run(prefix + args,cwd=target,env=env,capture_output=True,text=True,timeout=30)
        if result.returncode:
            raise RuntimeError("Worker Git initialization failed: " + result.stderr[-1000:])
