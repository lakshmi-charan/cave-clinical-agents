"""CAVE experiment driver.

Stages (run in this order; each is resumable and cached):
  python run.py check            # keys, FHIR server, model access
  python run.py pilot --n 2      # 2 tasks per category per agent + verification, prints projected cost
  python run.py base             # all 300 MedAgentBench tasks for every agent
  python run.py refsol-check     # official grader agreement (needs refsol.py; BEFORE stress-setup)
  python run.py stress-setup     # writes stress-suite resources to the FHIR server (irreversible)
  python run.py stress           # stress-suite tasks for every agent
  python run.py verify           # all verifier calls
  python run.py analyze          # features, policies, conformal analysis, figures, results.json
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path

from cave.common import cfg, load_env, out_dir, read_json, write_json, ROOT


def agents():
    return [a["name"] for a in cfg()["agents"]]


def all_tasks(include_stress=True):
    from cave import tasks
    ts = tasks.base_tasks()
    if include_stress and (out_dir("tasks") / "stress_tasks.json").exists():
        ts = ts + tasks.stress_tasks()
    return ts


def cmd_check(a):
    from cave import fhir, llm
    envf = load_env()
    print("env file:", envf or "NOT FOUND (create .env with OPENAI_API_KEY and ANTHROPIC_API_KEY)")
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        print(f"  {k}: {'set' if os.environ.get(k) else 'MISSING'}")
    ok = fhir.ping()
    print("FHIR server:", "OK" if ok else f"NOT REACHABLE at {fhir.base()}")
    if ok:
        pats = fhir.search_all("Patient", {"_count": 5})
        print("  sample patient ids:", [p.get("id") for p in pats[:5]])
        p = fhir.read("Patient", "S6534835")
        print("  Patient/S6534835 readable (id == MRN):", p is not None)
    models = sorted({m for x in cfg()["agents"] for m in (x.get("model", x["name"]), x["same_family"],
                                                           x["cross_family"], *x.get("panel", []))} | set(cfg()["panel"]))
    for m in models:
        try:
            r = llm.chat(m, [{"role": "user", "content": "Reply with the single word OK."}], max_tokens=5,
                         stage="check", salt="check")
            print(f"  model {m}: OK ({r['text'].strip()[:20]!r})")
        except Exception as e:
            print(f"  model {m}: FAILED -> {str(e)[:300]}")
    try:
        llm.embed(["ping"])
        print("  embeddings: OK")
    except Exception as e:
        print("  embeddings: FAILED ->", str(e)[:200])
    print("refsol.py:", "found" if _refsol_path() else "not found (optional; needed for refsol-check)")
    llm.flush_ledger()


def _run_agents(task_list, limit_note=""):
    from cave import agent, llm
    for ag in agents():
        agent.run_agent(ag, task_list, workers=cfg().get("workers_agent", 4))
        llm.flush_ledger()
        print(f"  cost so far: ${llm.ledger()['total_usd']:.3f}")


def cmd_pilot(a):
    from cave import tasks, verify, llm
    load_env()
    base = tasks.base_tasks()
    by = defaultdict(list)
    for t in base:
        by[t["id"].split("_")[0]].append(t)
    sub = [t for k in sorted(by) for t in by[k][:a.n]]
    before = llm.ledger()["total_usd"]
    before_p = dict(llm.spend_by_provider())
    _run_agents(sub)
    after_agents = llm.ledger()["total_usd"]
    tb = {t["id"]: t for t in base}
    for ag in agents():
        recs = [read_json(out_dir("agents", ag) / f"{t['id']}.json") for t in sub]
        verify.run_verify(ag, tb, recs, workers=cfg().get("workers_verify", 6))
    llm.flush_ledger()
    after = llm.ledger()["total_usd"]
    n = len(sub)
    n_stress = 4 * cfg()["stress"]["n_per_hazard"] + 3 * cfg()["stress"]["n_per_unfamiliar"]
    per_task = (after - before) / max(1, n)
    print(f"\nPilot: {n} tasks/agent. agent cost ${after_agents - before:.3f}, verify cost ${after - after_agents:.3f}")
    print(f"Projected total for {300 + n_stress} tasks/agent: ${per_task * (300 + n_stress):.2f}")
    after_p = llm.spend_by_provider()
    for prov in sorted(after_p):
        d = after_p[prov] - before_p.get(prov, 0.0)
        cap = (cfg().get("max_cost_by_provider") or {}).get(prov)
        print(f"  projected {prov}: ${d / max(1, n) * (300 + n_stress):.2f}" + (f" (cap ${cap})" if cap else ""))
    for ag in agents():
        recs = [read_json(out_dir("agents", ag) / f"{t['id']}.json") for t in sub]
        print(f"  {ag}: internal success {sum(r['success_internal'] for r in recs)}/{n}, "
              f"units {sum(len(r['units']) for r in recs)}")


def cmd_base(a):
    from cave import tasks
    load_env()
    tasks.scan()
    base = tasks.base_tasks()
    if a.limit:
        base = base[:a.limit]
    _run_agents(base)
    for ag in agents():
        recs = [read_json(out_dir("agents", ag) / f"{t['id']}.json") for t in base]
        recs = [r for r in recs if r]
        print(f"{ag}: internal success {sum(r['success_internal'] for r in recs)}/{len(recs)}")


def _refsol_path():
    for p in [ROOT / "refsol.py", ROOT.parent / "refsol.py", Path.cwd() / "refsol.py"]:
        if p.exists():
            return p
    return None


def _load_refsol():
    p = _refsol_path()
    if p is None:
        return None
    shim = out_dir("refsol_shim")
    pkg = shim / "refsol_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text((ROOT / "cave" / "mab_utils.py").read_text(encoding="utf-8"), encoding="utf-8")
    src = p.read_text(encoding="utf-8").replace("from src.server.tasks.medagentbench.utils import",
                                                "from .utils import")
    (pkg / "refsol.py").write_text(src, encoding="utf-8")
    sys.path.insert(0, str(shim))
    return importlib.import_module("refsol_pkg.refsol")


class _H:
    def __init__(self, role, content):
        self.role, self.content = role, content


def _as_taskoutput(rec):
    """Rebuild the AgentBench-style history the official grader expects (THOUGHT lines removed)."""
    from cave.agent import PROMPT
    hist = [_H("user", "(prompt)")]
    for s in rec["steps"]:
        raw = s["raw"]
        lines = raw.strip().replace("```", "").split("\n")
        k = next((i for i, l in enumerate(lines) if l.strip().startswith(("GET", "POST", "FINISH("))), 0)
        hist.append(_H("agent", "\n".join(lines[k:]).strip()))
        if s["kind"] == "GET":
            hist.append(_H("user", f"Here is the response from the GET request:\n{s.get('response')}. Please call "
                                   "FINISH if you have got answers for all the questions and finished all the "
                                   "requested tasks" if s.get("ok") else s.get("response", "")))
        elif s["kind"] == "POST":
            hist.append(_H("user", "POST request accepted and executed successfully. Please call FINISH if you have "
                                   "got answers for all the questions and finished all the requested tasks"
                           if s.get("ok") else "Invalid POST request"))
    fin = [s for s in rec["steps"] if s["kind"] == "FINISH"]
    return types.SimpleNamespace(result=fin[-1]["answer_raw"] if fin else None, history=hist)


def cmd_refsol(a):
    from cave import tasks
    if (out_dir("tasks") / "setup_done.json").exists():
        print("WARNING: stress setup already applied; official grading may be affected for modified patients.")
    rs = _load_refsol()
    if rs is None:
        print("refsol.py not found; place it in the project folder.")
        return
    data = {d["id"]: d for d in json.loads((ROOT / "cave" / "test_data_v2.json").read_text(encoding="utf-8"))}
    out = {}
    for ag in agents():
        agree = n = off = mine = 0
        per = {}
        for tid, d in data.items():
            rec = read_json(out_dir("agents", ag) / f"{tid}.json")
            if not rec:
                continue
            to = _as_taskoutput(rec)
            try:
                official = bool(to.result is not None and getattr(rs, tid.split("_")[0])(d, to, cfg()["fhir_api_base"])
                                is True)
            except Exception as e:
                official = False
            n += 1
            off += official
            mine += rec["success_internal"]
            agree += official == rec["success_internal"]
            per[tid] = {"official": official, "internal": rec["success_internal"]}
        out[ag] = {"n": n, "official_success": off / max(1, n), "internal_success": mine / max(1, n),
                   "agreement": agree / max(1, n), "per_task": per}
        print(f"{ag}: official {off}/{n}, internal {mine}/{n}, agreement {agree}/{n}")
    write_json(out_dir("results") / "refsol_check.json", out)


def cmd_stress_setup(a):
    from cave import tasks
    load_env()
    ts = tasks.stress_tasks(apply=True)
    from collections import Counter
    print("stress tasks:", Counter(t["stressor"] for t in ts))


def cmd_stress(a):
    from cave import tasks
    load_env()
    ts = tasks.stress_tasks()
    _run_agents(ts)


def cmd_reapply_stress(a):
    """Re-create the stress-suite resources after the EHR container was reset (same seeded plan)."""
    from cave import tasks
    from cave.common import read_json
    load_env()
    flag = out_dir("tasks") / "setup_done.json"
    if flag.exists():
        flag.unlink()
    plan_tasks, setup = tasks.build_stress()
    old = read_json(out_dir("tasks") / "stress_tasks.json")
    assert old is None or [t["id"] for t in old] == [t["id"] for t in plan_tasks], "stress plan changed"
    tasks.apply_setup(setup)
    # verify references are unchanged
    from cave.reference import solve
    bad = [t["id"] for t in old or [] if solve(t) != t["ref"]]
    print("stress resources re-applied;", "references identical" if not bad else f"REFERENCE MISMATCH: {bad[:10]}")


def cmd_reevidence(a):
    """Recompute the v2 evidence (E1+/E2+) for already-logged runs, against the current EHR state.
    --which base: base tasks (run on a freshly reset EHR); --which stress: stress tasks (after reapply-stress)."""
    from cave import evidence
    load_env()
    ts = all_tasks()
    ts = [t for t in ts if (t["stressor"] == "base") == (a.which == "base")]
    n = 0
    for ag in agents():
        d = out_dir("agents", ag)
        for t in ts:
            p = d / f"{t['id']}.json"
            rec = read_json(p)
            if not rec:
                continue
            if all("E1v2" in u for u in rec["units"]) and not a.force:
                continue
            for i, u in enumerate(rec["units"]):
                u["E1v2"] = evidence.e1v2(t, u, rec["units"][:i])
                u["E2v2"] = evidence.e2v2(t, u, rec["units"][:i])
            write_json(p, rec)
            n += 1
    print(f"re-evidenced {n} task records ({a.which})")


def cmd_features(a):
    """Build per-action feature tables (needs the embeddings API) without running the heavy analysis."""
    import pandas as pd
    from cave import features, llm
    load_env()
    ts = all_tasks()
    tb = {t["id"]: t for t in ts}
    for ag in agents():
        recs = [read_json(out_dir("agents", ag) / f"{t['id']}.json") for t in ts]
        recs = [r for r in recs if r]
        trows = []
        for r in recs:
            t = tb[r["task_id"]]
            trows.append({"task_id": r["task_id"], "stressor": r["stressor"], "template": r["template"],
                          "hazard": bool(t.get("hazard")), "success": int(r["success_internal"]),
                          "status": r["status"], "n_rounds": r["n_rounds"], "n_units": len(r["units"]),
                          "n_get": sum(1 for s in r["steps"] if s["kind"] == "GET"),
                          "n_post": sum(1 for s in r["steps"] if s["kind"] == "POST"),
                          "agent_cost": r["agent_cost"]})
        pd.DataFrame(trows).to_csv(out_dir("results") / f"tasks_{ag}.csv", index=False)
        df = features.build(ag, tb, recs)
        df.to_pickle(out_dir("results") / f"units_{ag}.pkl")
        print(f"[{ag}] features: {len(df)} actions")
    llm.flush_ledger()
    print(f"total spend: ${llm.ledger()['total_usd']:.2f}", llm.spend_by_provider())


def cmd_verify(a):
    from cave import verify, llm
    load_env()
    ts = all_tasks()
    tb = {t["id"]: t for t in ts}
    for ag in agents():
        recs = [read_json(out_dir("agents", ag) / f"{t['id']}.json") for t in ts]
        recs = [r for r in recs if r]
        verify.run_verify(ag, tb, recs, workers=cfg().get("workers_verify", 6))
        llm.flush_ledger()
        print(f"  cost so far: ${llm.ledger()['total_usd']:.3f}")


def cmd_analyze(a):
    import pandas as pd
    from cave import features, analysis, llm
    from cave.verify import roles_for
    load_env()
    ts = all_tasks()
    tb = {t["id"]: t for t in ts}
    res_all = {}
    for ag in agents():
        recs = [read_json(out_dir("agents", ag) / f"{t['id']}.json") for t in ts]
        recs = [r for r in recs if r]
        trows = []
        for r in recs:
            t = tb[r["task_id"]]
            trows.append({"task_id": r["task_id"], "stressor": r["stressor"], "template": r["template"],
                          "hazard": bool(t.get("hazard")), "success": int(r["success_internal"]),
                          "status": r["status"], "n_rounds": r["n_rounds"], "n_units": len(r["units"]),
                          "n_get": sum(1 for s in r["steps"] if s["kind"] == "GET"),
                          "n_post": sum(1 for s in r["steps"] if s["kind"] == "POST"),
                          "agent_cost": r["agent_cost"]})
        pd.DataFrame(trows).to_csv(out_dir("results") / f"tasks_{ag}.csv", index=False)
        df = features.build(ag, tb, recs)
        df.to_pickle(out_dir("results") / f"units_{ag}.pkl")
        df.drop(columns=[c for c in df.columns if c.startswith("cost[")]).to_csv(
            out_dir("results") / f"units_{ag}.csv", index=False)
        roles = roles_for(ag)
        df.attrs = {"same_model": roles["same_model"], "fam_model": roles["same_family"],
                    "cross_model": roles["cross_family"]}
        desc = analysis.descriptive(df, n_boot=a.boot)
        raw = analysis.run(df, n_splits=a.splits)
        summ = analysis.summarize(raw)
        task_stats = {"n_tasks": len(recs), "internal_success_base": float(
            sum(r["success_internal"] for r in recs if r["stressor"] == "base") /
            max(1, sum(1 for r in recs if r["stressor"] == "base"))),
            "agent_cost_usd": float(sum(r["agent_cost"] for r in recs)),
            "status": pd.Series([r["status"] for r in recs]).value_counts().to_dict()}
        res_all[ag] = {"descriptive": desc, "summary": summ, "tasks": task_stats}
        write_json(out_dir("results") / f"results_{ag}.json", res_all[ag])
        print(f"[{ag}] analysis done: {len(df)} actions")
    res_all["cost_ledger"] = llm.ledger()
    write_json(out_dir("results") / "results_all.json", res_all)
    from cave import figures
    figures.make_all(res_all)
    print("results in", out_dir("results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "pilot", "base", "refsol-check", "stress-setup", "stress", "verify",
                                      "analyze", "reapply-stress", "features", "reevidence"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--which", choices=["base", "stress"], default="base")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    load_env()
    {"check": cmd_check, "pilot": cmd_pilot, "base": cmd_base, "refsol-check": cmd_refsol,
     "stress-setup": cmd_stress_setup, "stress": cmd_stress, "verify": cmd_verify, "analyze": cmd_analyze,
     "reapply-stress": cmd_reapply_stress, "features": cmd_features, "reevidence": cmd_reevidence}[a.stage](a)


if __name__ == "__main__":
    main()
