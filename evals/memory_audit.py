"""Deterministic retrieval diagnostics; never infer usefulness from a hit alone."""
from __future__ import annotations

import json


def audit_retrieval(messages, necessary_keys=(), stale_keys=()):
    searches = []
    by_id = {}
    seen_queries = set()
    seen_records = set()
    found = set()
    stale_found = set()
    for message in messages:
        if message.get("role") == "assistant":
            for block in message.get("content",[]):
                if block.get("type") != "toolCall" or block.get("name") != "memory_search":
                    continue
                args = block.get("arguments",{})
                signature = json.dumps({"query":" ".join(str(args.get("query","")).casefold().split()),
                    "kind":args.get("kind"),"limit":args.get("limit",3)},sort_keys=True)
                search = {"id":block["id"],"query":args.get("query"),"kind":args.get("kind"),
                    "duplicate_query":signature in seen_queries,"result_count":None,
                    "necessary_keys":[],"stale_keys":[],"repeated_records":0,"status":"missing_result"}
                seen_queries.add(signature)
                searches.append(search)
                by_id[block["id"]] = search
        elif message.get("role") == "toolResult" and message.get("tool_call_id") in by_id:
            search = by_id[message["tool_call_id"]]
            text = "".join(b.get("text","") for b in message.get("content",[]) if b.get("type") == "text")
            if message.get("is_error"):
                search["status"] = "suppressed" if "Memory search suppressed:" in text else "tool_error"
                continue
            try:
                values = json.loads(text)
            except (ValueError,TypeError):
                search["status"] = "invalid_result"
                continue
            if not isinstance(values,list) or any(not isinstance(v,dict) or "key" not in v for v in values):
                search["status"] = "invalid_result"
                continue
            search["status"] = "ok"
            search["result_count"] = len(values)
            keys = {v["key"] for v in values}
            search["necessary_keys"] = sorted(keys & set(necessary_keys))
            search["stale_keys"] = sorted(keys & set(stale_keys))
            found.update(search["necessary_keys"])
            stale_found.update(search["stale_keys"])
            for item in values:
                identity = json.dumps(item,sort_keys=True,ensure_ascii=False)
                search["repeated_records"] += int(identity in seen_records)
                seen_records.add(identity)
    valid = [s for s in searches if s["status"] == "ok"]
    return {"searches":searches,"search_count":len(searches),
        "successful_searches":len(valid),"empty_searches":sum(s["result_count"] == 0 for s in valid),
        "duplicate_queries":sum(s["duplicate_query"] for s in searches),
        "repeated_records":sum(s["repeated_records"] for s in searches),
        "suppressed_searches":sum(s["status"] == "suppressed" for s in searches),
        "unresolved_searches":sum(s["status"] != "ok" for s in searches),
        "necessary_keys_found":sorted(found),"stale_keys_returned":sorted(stale_found),
        "necessary_evidence_recall":len(found)/len(set(necessary_keys)) if necessary_keys else None,
        "limitations":"Exact normalized query/filters and exact repeated records only; not semantic duplication. Gold-key recall does not prove model adoption or causality. Stale results do not prove stale behavior."}
