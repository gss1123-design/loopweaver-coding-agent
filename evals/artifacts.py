"""Versioned run artifacts, independent of model scores and telemetry."""
from __future__ import annotations

import hashlib
import math
import json
from pathlib import Path
import uuid
from typing import Any


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


class ArtifactStore:
    def __init__(self, root: str | Path, config: dict[str, Any]) -> None:
        self.root = Path(root) / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.config = config
        self.write("manifest.json", {"schema_version": 1, "config": config, "config_hash": canonical_hash(config)})

    def write(self, relative: str, value: Any) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def record(self, record: dict[str, Any], *, events: list | None = None, messages: list | None = None) -> None:
        run_id = uuid.uuid4().hex
        directory = f"runs/{run_id}"
        self.write(f"{directory}/result.json", record)
        for name, values in (("events", events), ("session", messages)):
            if values is not None:
                target = self.root / directory / f"{name}.jsonl"
                target.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
        index = {**record, "run_id": run_id, "artifact_dir": directory}
        with (self.root / "runs.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(index, ensure_ascii=False) + "\n")


def compare_paired(records: list[dict[str, Any]], baseline: str, candidate: str) -> dict[str, Any]:
    """Missing, duplicated, or failed observations never become zero scores."""
    if baseline == candidate:
        raise ValueError("Baseline and candidate must differ")
    groups: dict[tuple, dict[str, list]] = {}
    for item in records:
        key = (item["layer"], item["case_id"], item["input_hash"], item["repetition"],
               item.get("benchmark_version"),item.get("model"),item.get("comparison_config"))
        groups.setdefault(key, {}).setdefault(item["harness_id"], []).append(item)
    pairs = []
    diagnostics = []
    for key, observations in groups.items():
        left, right = observations.get(baseline, []), observations.get(candidate, [])
        if len(left) != 1 or len(right) != 1:
            diagnostics.append({"key": key, "reason": "missing_or_duplicate_observation"})
            continue
        a, b = left[0], right[0]
        if a.get("outcome") != "scored" or b.get("outcome") != "scored":
            diagnostics.append({"key": key, "reason": "unscored_pair"})
            continue
        pairs.append((a, b))
    def delta(field: str, container: str) -> dict:
        eligible = [(a.get(container, {}).get(field), b.get(container, {}).get(field)) for a, b in pairs]
        eligible = [(a, b) for a, b in eligible if isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b)]
        return {"eligible_pairs": len(eligible), "mean_delta": sum(b-a for a,b in eligible)/len(eligible) if eligible else None}
    return {"paired_runs": len(pairs), "correctness": delta("correctness", "scores"),
            "tokens": delta("tokens", "telemetry"), "duration_ms": delta("duration_ms", "telemetry"),
            "cost": delta("cost", "telemetry"), "diagnostics": diagnostics}
