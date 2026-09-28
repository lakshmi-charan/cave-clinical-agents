"""Assemble one row per consequential action with labels, verifier outputs, signals and costs."""
import json
import math
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .common import cfg, out_dir, read_json
from .llm import embed
from .tasks import BASE_FUNCS
from .verify import roles_for

GROUP = {"base": "base", "S1_alias": "unfamiliar", "S2_drift": "unfamiliar", "S3_novel": "unfamiliar",
         "S4_allergy": "hazard", "S5_renal": "hazard", "S6_duplicate": "hazard", "S7_lookalike": "hazard"}
TIER_W = {0: 0.2, 1: 0.4, 2: 1.0}


def _cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _tool_name_for(step, funcs):
    if step["kind"] not in ("GET", "POST"):
        return None
    if step["kind"] == "GET":
        path = urlparse(step.get("url", "")).path.rstrip("/").split("/")[-1]
        cands = [f for f in funcs if f["name"].startswith("GET") and f["name"].rstrip("/").endswith("/" + path)]
    else:
        first = step["raw"].split("\n")
        first = next((l for l in first if l.strip().startswith("POST")), "")
        path = first.split()[-1].rstrip("/").split("/")[-1] if first.split() else ""
        cands = [f for f in funcs if f["name"].startswith("POST") and f["name"].rstrip("/").endswith("/" + path)]
    return cands[0] if cands else None


def _tool_text(f):
    return f["name"].replace("{api_base}", "") + " | " + f["description"][:300] + " | " + json.dumps(
        list(((f.get("parameters") or {}).get("properties") or {}).keys()))


def build(agent, tasks_by_id, recs):
    roles = roles_for(agent)
    same, fam, cross = roles["same_model"], roles["same_family"], roles["cross_family"]
    base_tool_vecs = embed([_tool_text(f) for f in BASE_FUNCS])
    base_shapes = set()
    for r in recs:
        if r["stressor"] == "base":
            for s in r["steps"]:
                for k in s.get("shape") or []:
                    base_shapes.add(k)
    scan = read_json(out_dir("tasks") / "scan.json", {"lab_values": {}})
    dist = {k: np.sort(np.asarray(v, dtype=float)) for k, v in scan.get("lab_values", {}).items() if v}

    rows, texts = [], []
    for r in recs:
        task = tasks_by_id[r["task_id"]]
        ver = read_json(out_dir("verify", agent) / f"{r['task_id']}.json", {})
        for ui, u in enumerate(r["units"]):
            v = ver.get(str(ui), {})
            row = {"agent": agent, "task_id": r["task_id"], "stressor": r["stressor"], "group": GROUP[r["stressor"]],
                   "template": r["template"], "hazard": bool(task.get("hazard")), "unit": ui, "type": u["type"],
                   "resource": u["resource"] or "FINISH", "tier": u["tier"], "w": TIER_W[u["tier"]],
                   "correct": int(bool(u["correct"])), "why": u.get("why"), "task_success": int(r["success_internal"]),
                   "E1": u["E1"]["verdict"], "E2": u["E2"]["verdict"], "E1_q": u["E1"]["queries"],
                   "E2_q": u["E2"]["queries"],
                   "E1v2": (u.get("E1v2") or {}).get("verdict", "missing"),
                   "E2v2": (u.get("E2v2") or {}).get("verdict", "missing"),
                   "E1v2_q": (u.get("E1v2") or {}).get("queries", 0), "E2v2_q": (u.get("E2v2") or {}).get("queries", 0), "n_prior_steps": u["step"],
                   "n_failed_gets": sum(1 for s in r["steps"][:u["step"]] if s["kind"] == "GET" and not s.get("ok"))}
            for key, val in v.items():
                row[f"dec[{key}]"] = int(val["decision"] == "approve")
                row[f"conf[{key}]"] = val["confidence"] / 100.0
                row[f"cost[{key}]"] = val.get("cost", 0.0)
            # role-based convenience columns
            for tag, m in [("same", same), ("fam", fam), ("cross", cross)]:
                row[f"v_{tag}"] = row.get(f"dec[std|{m}]", np.nan)
                row[f"c_{tag}"] = row.get(f"conf[std|{m}]", np.nan)
                row[f"b_{tag}"] = row.get(f"dec[blind|{m}]", np.nan)
            pflips = [row[f"dec[{p}|{same}]"] != row["v_same"] for p in ("p1", "p2") if f"dec[{p}|{same}]" in row]
            row["perturb_instab"] = float(np.mean(pflips)) if pflips else 0.0
            row["blind_flip"] = int(row.get(f"dec[blind|{same}]", row["v_same"]) != row["v_same"])
            # case rarity from the key lab value in the chart
            vals = [x for x in (task["ref"].get("facts") or {}).items() if x[1] is not None]
            rar = 0.0
            if vals:
                code, val = vals[0]
                if code in dist and len(dist[code]):
                    pct = np.searchsorted(dist[code], val) / len(dist[code])
                    rar = abs(pct - 0.5) * 2
            row["rarity"] = rar
            # tool novelty: tools used so far incl. this action
            used = []
            for s in r["steps"][:u["step"] + 1]:
                f = _tool_name_for(s, task["funcs"])
                if f is not None and f not in used:
                    used.append(f)
            row["_tools"] = [_tool_text(f) for f in used]
            shapes = set()
            for s in r["steps"][:u["step"]]:
                shapes |= set(s.get("shape") or [])
            row["shape_novelty"] = (len(shapes - base_shapes) / len(shapes)) if shapes else 0.0
            just = v.get(f"std|{same}", {}).get("justification", "")
            row["_thought"], row["_just"] = u.get("thought") or "", just
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # embeddings: rationale echo and tool novelty
    th = embed(list(df["_thought"].replace("", " ")))
    ju = embed(list(df["_just"].replace("", " ")))
    df["echo"] = [_cos(a, b) for a, b in zip(th, ju)]
    all_tools = sorted({t for ts in df["_tools"] for t in ts})
    tv = dict(zip(all_tools, embed(all_tools))) if all_tools else {}
    nov = []
    for ts in df["_tools"]:
        n = 0.0
        for t in ts:
            n = max(n, 1 - max(_cos(tv[t], b) for b in base_tool_vecs))
        nov.append(n)
    df["tool_novelty"] = nov
    df = df.drop(columns=["_tools", "_thought", "_just"])
    df["SF_same"] = ((df["correct"] == 0) & (df["v_same"] == 1)).astype(int)
    return df
