"""Audit saved live runs; no model calls and no guessed token attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import ArtifactStore
from .memory_audit import audit_retrieval


def analyze(report_path: Path) -> dict:
    report=json.loads(report_path.read_text(encoding="utf-8"))
    index=[json.loads(line) for line in (report_path.parent/"runs.jsonl").read_text(encoding="utf-8").splitlines()]
    observations=[]
    for record in report["records"]:
        matches=[r for r in index if (r["case_id"],r["repetition"],r["harness_id"]) == (record["case_id"],record["repetition"],record["harness_id"])]
        if len(matches) != 1:
            raise ValueError("Missing or duplicated artifact for observation")
        path=report_path.parent/matches[0]["artifact_dir"]/"session.jsonl"
        messages=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        requests=[]
        searches=[]
        calls={}
        notes_seen=False
        for message in messages:
            if message["role"] == "assistant":
                usage=message.get("usage",{})
                request={"input_tokens":usage.get("input"),"output_tokens":usage.get("output"),"calls":[]}
                for block in message["content"]:
                    if block["type"] != "toolCall":
                        continue
                    calls[block["id"]]=block
                    request["calls"].append({"name":block["name"],"args":block["arguments"]})
                    if block["name"] == "memory_search":
                        searches.append({"id":block["id"],"query":block["arguments"].get("query"),"after_notes_read":notes_seen})
                requests.append(request)
            elif message["role"] == "toolResult":
                call=calls.get(message["tool_call_id"],{})
                if call.get("name") == "read" and call.get("arguments",{}).get("path") == "PROJECT_NOTES.md" and not message.get("is_error"):
                    notes_seen=True
                if message["tool_name"] == "memory_search":
                    text="".join(b.get("text","") for b in message["content"] if b["type"] == "text")
                    search=next(s for s in searches if s["id"] == message["tool_call_id"])
                    search["result_chars"]=len(text)
                    try:
                        values=json.loads(text)
                        search["result_count"]=len(values) if isinstance(values,list) else None
                    except ValueError:
                        search["result_count"]=None
        observations.append({"case":record["case_id"],"rep":record["repetition"],"arm":record["harness_id"],
            "requests":requests,"searches":searches,"scores":record["scores"],"outcome":record["outcome"],"artifact":str(path),
            "retrieval_audit":audit_retrieval(messages,record.get("inputs",{}).get("necessary_keys",()),record.get("inputs",{}).get("stale_keys",()))})
    totals={}
    for arm in dict.fromkeys(("memory-off","memory-on",*(r["arm"] for r in observations))):
        runs=[r for r in observations if r["arm"] == arm]
        totals[arm]={"runs":len(runs),"model_requests":sum(len(r["requests"]) for r in runs),
            "input_tokens":sum(q["input_tokens"] or 0 for r in runs for q in r["requests"]),
            "output_tokens":sum(q["output_tokens"] or 0 for r in runs for q in r["requests"]),
            "searches":sum(len(r["searches"]) for r in runs),
            "searches_after_notes":sum(s["after_notes_read"] for r in runs for s in r["searches"]),
            "nonempty_searches":sum((s.get("result_count") or 0)>0 for r in runs for s in r["searches"]),
            **{field:sum(r["retrieval_audit"][field] for r in runs) for field in ("empty_searches","duplicate_queries","repeated_records","suppressed_searches","unresolved_searches")}}
    return {"source_report":str(report_path),"totals":totals,"observations":observations,
        "limitations":"API token totals are observed. Individual schema/result/repeated-history costs are not measured separately. Retrieval hit is not evidence of useful information."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--artifacts",default="output/evals-analysis")
    args=parser.parse_args()
    result=analyze(args.report)
    store=ArtifactStore(args.artifacts,{"mode":"offline-live-trace-audit","source":str(args.report)})
    store.write("analysis.json",result)
    print(json.dumps({"artifact_dir":str(store.root),"totals":result["totals"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
