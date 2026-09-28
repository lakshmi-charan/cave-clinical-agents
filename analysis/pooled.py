"""Pooled policy analysis per agent model (Sections 5.2-5.5 of the paper).

For each model and evidence version it runs 200 random task-level splits (development / calibration / test)
and evaluates every gating policy on identical actions.

usage:  python analysis/pooled.py MODEL EV [--alpha 0.02] [--tag main] [--splits 200]
        EV is v2 (strengthened evidence E1+/E2+) or v1 (pre-specified evidence E1/E2)
output: runs/analysis/pooled/<MODEL>/<EV>_<TAG>.json  (+ <EV>_budget.json and <EV>_raw.json for tag "main")
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import MODELS, OUT, load_model
from cave import analysis
from cave.common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("ev", choices=["v1", "v2"])
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()

    df, T = load_model(a.model, a.ev)
    out = OUT / "pooled" / a.model
    out.mkdir(parents=True, exist_ok=True)
    panel = analysis.panel_models(df)
    desc = analysis.descriptive(df, n_boot=1000)
    raw = analysis.run(df, n_splits=a.splits, main_alpha=a.alpha)
    res = {"descriptive": desc, "summary": analysis.summarize(raw), "runs": MODELS[a.model], "panel": panel,
           "ev": a.ev, "alpha": a.alpha, "n_units": int(len(df)), "n_tasks": int(len(T))}
    write_json(out / f"{a.ev}_{a.tag}.json", res)

    if a.tag == "main":
        # fixed escalation budgets: CAVE ranking vs. confidence vs. random (Table 12)
        cv = pd.DataFrame(raw["curves"])
        budget = {}
        for fr in [0.1, 0.2, 0.3, 0.5]:
            d = cv[np.isclose(cv.frac, fr)]
            for b in ["random", "confidence"]:
                diff = (d.CAVE - d[b]) * 100
                budget[f"{fr}|{b}"] = [diff.mean(), np.percentile(diff, 2.5), np.percentile(diff, 97.5),
                                       (diff < 0).mean(), d.CAVE.mean() / d[b].mean() - 1]
            budget[f"{fr}|abs"] = [d.CAVE.mean() * 100, d.confidence.mean() * 100, d.random.mean() * 100]
        json.dump(budget, open(out / f"{a.ev}_budget.json", "w"), indent=1)
        json.dump({"curves": raw["curves"], "auc": raw["auc"], "guarantee": raw["guarantee"]},
                  open(out / f"{a.ev}_raw.json", "w"), default=lambda o: None)
    print("done", a.model, a.ev, a.tag, len(df), "actions")


if __name__ == "__main__":
    main()
