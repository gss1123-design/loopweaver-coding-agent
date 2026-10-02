"""Fixture contracts, shared without importing the paid-model runner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEvidence:
    key: str
    value: str
    kind: str = "decision"
    source: str = "fixture:previous-session"
    updated: float = 1700000000.0
    source_check: str | None = None


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    source: str
    notes: str
    memory: str
    checks: str
    retrieval_needed: bool | None = False
    abstention_checks: str | None = None
    category: str = "current_documents"
    memories: tuple[MemoryEvidence, ...] | None = None
    necessary_keys: tuple[str, ...] = ()
    stale_keys: tuple[str, ...] = ()
    stale_checks: str | None = None
    misattribution_checks: str | None = None
    witness_message: str | None = None
    witness_session_id: str | None = None
    witness_entry_id: str | None = None
