"""Shared helpers for the paper's analysis: run lists, verifier roles and data loading.

The inputs are the per-action tables written by `python run.py features` (runs/results/units_<run>.pkl and
tasks_<run>.csv) and the clinical labels written by `python scripts/relabel.py <run>`.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CAVE_CONFIG", str(ROOT / "config.yaml"))

import pandas as pd  # noqa: E402

from cave.common import cfg  # noqa: E402

RESULTS = ROOT / "runs" / "results"
OUT = ROOT / "runs" / "analysis"

# Agent models and their runs, in the order used throughout the paper.
MODELS = {
    "gpt-5-mini": ["gpt-5-mini", "gpt-5-mini__r2", "gpt-5-mini__r3"],
    "gpt-5": ["gpt-5", "gpt-5__r2"],
    "claude-haiku-4-5": ["claude-haiku-4-5"],
}
LABEL = {"gpt-5-mini": "gpt-5-mini", "gpt-5": "GPT-5", "claude-haiku-4-5": "Claude Haiku 4.5"}
# Runs that did not exist when the strengthened evidence (E1+/E2+) was designed.
HELD_OUT = {"gpt-5-mini__r2", "gpt-5-mini__r3", "gpt-5__r2", "claude-haiku-4-5"}


def roles(model):
    """(same model, same family, cross family) verifiers of an agent, as configured in config.yaml."""
    a = next(x for x in cfg()["agents"] if x["name"] == model)
    return a.get("model", a["name"]), a["same_family"], a["cross_family"]


def load_run(run):
    """Per-action table of one run with clinical (semantic) labels; strict labels kept as `correct_strict`."""
    U = pd.read_pickle(RESULTS / f"units_{run}.pkl")
    T = pd.read_csv(RESULTS / f"tasks_{run}.csv")
    L = pd.read_csv(RESULTS / f"labels_semantic_{run}.csv")
    TS = pd.read_csv(RESULTS / f"tasks_semantic_{run}.csv")
    m = U.merge(L[["task_id", "unit", "correct_strict", "correct_sem"]], on=["task_id", "unit"], how="left")
    if m.correct_sem.isna().any() or not (m.correct_strict == m.correct).all():
        raise ValueError(f"{run}: labels do not match the logged actions; re-run scripts/relabel.py {run}")
    success = dict(zip(TS.task_id, TS.success_sem))
    m["correct"] = m.correct_sem.astype(int)
    m["task_success"] = m.task_id.map(success).astype(int)
    m = m.drop(columns=["correct_sem"])
    m.attrs = {}
    T = T.merge(TS, on="task_id")
    T["success_strict"], T["success"] = T["success"], T["success_sem"]
    return m, T


def load_model(model, ev="v2"):
    """Pooled per-action table of all runs of one agent model.

    ev="v2" uses the strengthened evidence (E1+/E2+), ev="v1" the pre-specified evidence (E1/E2).
    Repeat runs of the same task share a `cluster`, so splits and bootstraps keep them together."""
    Us, Ts = [], []
    for r in MODELS[model]:
        U, T = load_run(r)
        U["cluster"], U["run"], U["task_id"] = U["task_id"], r, r + ":" + U["task_id"]
        T["run"] = r
        Us.append(U)
        Ts.append(T)
    df, T = pd.concat(Us, ignore_index=True), pd.concat(Ts, ignore_index=True)
    if ev == "v2":
        df["E1"], df["E2"], df["E1_q"], df["E2_q"] = df["E1v2"], df["E2v2"], df["E1v2_q"], df["E2v2_q"]
    same, fam, cross = roles(model)
    df.attrs = {"same_model": same, "fam_model": fam, "cross_model": cross}
    return df, T
