"""Validate authored fixture/check contracts in Docker, with zero model calls."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

from .artifacts import ArtifactStore
from .live_memory import BENCHMARK_VERSION,TASKS,HISTORY_TASK,VERIFIED_HISTORY_TASK,grade,seed_witness,task_evidence
from .memory_cases import EXTENDED_TASKS
from .memory_scale_cases import SCALED_TASKS,DENSITY_TASKS,WITNESSED_SPOOF_TASKS,NEUTRAL_ABLATION_TASKS
from coding_agent.memory import MemoryStore


# Hidden reference implementations verify graders, not model capability.
SOLUTIONS = {
    "dual_history_preferences":"def delivery_defaults():\n    return {'format':'compact-json','timezone':'Asia/Shanghai'}\n",
    "slug_policy":"def slug(text):\n    return '-'.join(text.strip().lower().split()) or 'untitled'\n",
    "pagination_update":"def paginate(items,page,size):\n    if page < 1 or size < 1: raise ValueError('invalid pagination')\n    return items[(page-1)*size:page*size]\n",
    "unknown_config":"def deployment_config():\n    return {'region':'ap-southeast-1','backup_bucket':None}\n",
    "historical_release":"def release_policy():\n    return {'batch_size':17,'release_tag':'harbor-47'}\n",
    "verified_historical_release":"def release_policy():\n    return {'batch_size':17,'release_tag':'harbor-47'}\n",
    "current_retry_limit":"def retry_limit():\n    return 2\n",
    "historical_style":"def output_style():\n    return 'compact-json'\n",
    "historical_locale":"def notification_locale():\n    return 'zh-CN'\n",
    "conflicting_release":"def release_channel():\n    return 'stable'\n",
    "conflicting_timeout":"def retry_timeout():\n    return 8\n",
    "missing_retention":"def retention_days():\n    return None\n",
    "missing_owner":"def escalation_owner():\n    return None\n",
    "scaled_unverified_source":"def rollout_gate():\n    return None\n",
    "scaled_registered_source":"def rollout_gate():\n    return 'review-required'\n",
    "scaled_freshness_equal_trust":"def rollout_gate():\n    return 'review-required'\n",
    "scaled_registered_source_16":"def rollout_gate():\n    return 'review-required'\n",
    "scaled_registered_source_160":"def rollout_gate():\n    return 'review-required'\n",
    "spoofed_registered_id_with_witness":"def rollout_gate():\n    return 'review-required'\n",
    "spoofed_registered_id_reversed_order":"def rollout_gate():\n    return 'review-required'\n",
    "neutral_source_ablation_a":"def rollout_gate():\n    return 'phase-n4'\n",
    "neutral_source_ablation_b":"def rollout_gate():\n    return 'phase-n4'\n",
}


def validate(store,image):
    observations=[]
    for task in (*TASKS,HISTORY_TASK,VERIFIED_HISTORY_TASK,*EXTENDED_TASKS,*SCALED_TASKS,*DENSITY_TASKS,*WITNESSED_SPOOF_TASKS,*NEUTRAL_ABLATION_TASKS):
        workspace = store.root / "workspaces" / task.name
        workspace.mkdir(parents=True)
        (workspace/"PROJECT_NOTES.md").write_text(task.notes,encoding="utf-8")
        (workspace/"app.py").write_text(task.source,encoding="utf-8")
        seed_witness(task, workspace)
        provenance_valid = True
        if task.witness_message is not None:
            memory_store = MemoryStore(workspace)
            for row in task_evidence(task)[-2:]:
                if row["source_check"] == "matched_user_entry":
                    _,session_id,entry_id = row["source"].split(":",2)
                    memory_store.put_confirmed_user_quote("workspace",row["key"],row["value"],
                        kind=row["kind"],session_id=session_id,entry_id=entry_id)
                else:
                    memory_store._put("workspace",row["key"],row["value"],kind=row["kind"],
                        source=row["source"],allow_session_source=True)
            actual = {hit["key"]:hit.get("source_check")
                for row in task_evidence(task)[-2:]
                for hit in memory_store.search("workspace",row["key"])}
            provenance_valid = all(actual.get(row["key"]) == row["source_check"] for row in task_evidence(task)[-2:])
        baseline = grade(workspace,task,image)
        baseline_valid = baseline.get("exit_code") == 1 and "AssertionError" in baseline.get("stderr","")
        (workspace/"app.py").write_text(SOLUTIONS[task.name],encoding="utf-8")
        reference = grade(workspace,task,image)
        alternatives = {}
        for name,checks,expected in (("stale",task.stale_checks,False),
            ("abstention",task.abstention_checks,task.category in {"no_evidence","source_authenticity_unverified"}),
            ("misattribution",task.misattribution_checks,False)):
            if checks:
                observed = grade(workspace,replace(task,checks=checks),image)
                alternatives[name] = {"expected_pass":expected,"verification":observed,
                    "valid":observed["passed"] == expected and observed.get("exit_code") in (0,1)
                        and (observed["passed"] or "AssertionError" in observed.get("stderr","") )}
        valid = baseline_valid and reference["passed"] and provenance_valid and all(v["valid"] for v in alternatives.values())
        record = {"case_id":task.name,"category":task.category,"valid":valid,
            "baseline":baseline,"reference":reference,"alternatives":alternatives,
            "provenance_valid":provenance_valid}
        observations.append(record)
        store.write("fixture-validation.json",{"benchmark_version":BENCHMARK_VERSION,"api_calls":0,"observations":observations,
            "all_valid":all(r["valid"] for r in observations)})
        print(json.dumps({"case":task.name,"valid":valid}),flush=True)
        if not valid:
            break  # Do not continue a broken verifier or invalid fixture.
    return observations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",default="xingclaw-sandbox:local")
    parser.add_argument("--artifacts",default="output/evals-fixtures")
    args=parser.parse_args()
    store=ArtifactStore(args.artifacts,{"mode":"offline-fixture-validation","benchmark_version":BENCHMARK_VERSION,"api_calls":0,"image":args.image})
    records=validate(store,args.image)
    print(json.dumps({"artifact_dir":str(store.root),"validated":len(records),"api_calls":0},indent=2))
    return int(len(records) != len(SOLUTIONS) or not all(r["valid"] for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
