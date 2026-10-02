from __future__ import annotations

import re
from pathlib import Path

from .types import LoadedExtensions, RegisteredCommand, SkillSpec


def discover_skill_paths(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> list[Path]:
    workspace = Path(workspace_dir)
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        paths.append(resolved)

    default_dir = workspace / ".xingclaw" / "skills"
    if default_dir.exists() and default_dir.is_dir():
        for path in sorted(default_dir.glob("*.md")):
            _add(path)

    for raw in configured_paths or []:
        target = Path(raw)
        if not target.is_absolute():
            target = workspace / raw
        if target.exists() and target.is_dir():
            for path in sorted(target.glob("*.md")):
                _add(path)
        elif target.exists() and target.is_file() and target.suffix.lower() == ".md":
            _add(target)

    return paths


def load_skills(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> LoadedExtensions:
    """Load the skill catalog without putting skill bodies in the prompt.

    We still read each file once here so frontmatter and a fallback Markdown
    title can be parsed.  The body is deliberately discarded after metadata
    extraction; ``load_skill_content`` reads it again only after a command or
    the model-facing ``use_skill`` tool explicitly selects that skill.
    """

    result = LoadedExtensions()
    seen_cmds: dict[str, str] = {}
    for path in discover_skill_paths(workspace_dir, configured_paths=configured_paths):
        try:
            raw_text = path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            meta, text = _parse_skill_frontmatter(raw_text)
            title = str(meta.get("name") or _extract_title(text) or path.stem).strip()
            if not title:
                title = path.stem
            cmd = str(meta.get("command") or f"skill:{_slugify(title)}").strip().lstrip("/")
            if not cmd:
                cmd = f"skill:{_slugify(path.stem)}"
            desc = str(meta.get("description") or f"执行技能：{title}").strip()
            skill = SkillSpec(
                name=title,
                command_name=cmd,
                description=desc,
                content="",
                source_path=str(path),
            )
            if cmd in seen_cmds:
                result.diagnostics.append(f"skill command conflict: /{cmd} from {path} overrides {seen_cmds[cmd]}")
            seen_cmds[cmd] = str(path)

            result.skills.append(skill)
            # Only the catalog entry is always visible.  The potentially long
            # procedure is loaded lazily by load_skill_content().
            result.append_prompts.append(
                f"## Available Skill: {title}\n"
                f"- command: /{cmd}\n"
                f"- description: {desc}\n"
                "- full instructions: load only when this skill is relevant "
                "(use the use_skill tool or the slash command)."
            )
            result.commands[cmd] = RegisteredCommand(
                name=cmd,
                description=desc,
                source="skill",
                handler=lambda ctx, _skill=skill: _render_skill_prompt(_skill, ctx.raw_text),
                skill=skill,
            )
            result.loaded_paths.append(str(path))
        except Exception as exc:
            result.errors.append(f"{path}: {exc}")
    return result


def load_skill_content(skill: SkillSpec) -> str:
    """Read and return one skill body at activation time.

    The path comes from the discovered workspace catalog rather than user
    input, so callers cannot use this helper to read an arbitrary file.  The
    file is re-read on activation, which also lets a long-running session pick
    up an edited skill without rebuilding the entire session.
    """

    path = Path(skill.source_path)
    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Unable to load skill {skill.name!r}: {exc}") from exc
    if not raw_text:
        raise ValueError(f"Skill {skill.name!r} is empty")
    _, content = _parse_skill_frontmatter(raw_text)
    if not content.strip():
        raise ValueError(f"Skill {skill.name!r} has no instructions")
    return content.strip()


def resolve_skill(skills: list[SkillSpec], identifier: str) -> SkillSpec | None:
    """Resolve a skill by command, name, or filename stem.

    Command conflicts follow the same last-definition-wins behavior as the
    command registry, hence the reverse scan for deterministic resolution.
    """

    key = str(identifier or "").strip().lstrip("/").casefold()
    if not key:
        return None
    command_key = key if key.startswith("skill:") else f"skill:{key}"
    for skill in reversed(skills):
        command = skill.command_name.casefold()
        stem = Path(skill.source_path).stem.casefold()
        name = skill.name.casefold()
        if key in {command, stem, name} or command_key == command:
            return skill
    return None


def _extract_title(text: str) -> str | None:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    return None


def _parse_skill_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip().lower()
        val = v.strip().strip("'").strip('"')
        if key and val:
            meta[key] = val
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "skill"


def _render_skill_prompt(skill: SkillSpec, raw_text: str) -> str:
    cmd_text = raw_text.strip() if raw_text else f"/{skill.command_name}"
    return (
        f"请执行技能 `{skill.name}`（命令：`{cmd_text}`）。\n"
        "请按照该技能的完整流程处理当前请求，并直接给出可执行结果。"
    )
