"""Formal tests and task-level conformal calibration (Section 5.6 of the paper). No model calls.

(A) Cross-fitted CAVE risk scores (10 folds grouped by task) and paired task-cluster bootstrap tests
    (2,000 resamples) with Holm correction over all comparisons.
(B) Task-level conformal risk control, whose guarantee needs exchangeable tasks only.

usage:  python analysis/stats.py EV        (EV = v2 or v1)
output: runs/analysis/stats/stats_<EV>.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import MODELS, OUT, load_model
from cave import analysis as A

N_BOOT, N_SPLIT, BUDGET = 2000, 200, 0.2


def crossfit(df, k=10, seed=0):
    idf = df[df.group != "unfamiliar"]
    cl = np.random.default_rng(seed).permutation(idf.cluster.unique())
    fold = {c: i % k for i, c in enumerate(cl)}
    s = pd.Series(np.nan, index=df.index)
    for f in range(k):
        te = idf[idf.cluster.map(fold) == f]; tr = idf[idf.cluster.map(fold) != f]
        s.loc[te.index] = A.predict(A.fit_estimator(tr, A.FEATS), te)
    sh = df[df.group == "unfamiliar"]
    if len(sh):
        s.loc[sh.index] = A.predict(A.fit_estimator(idf, A.FEATS), sh)
    return s.values


def stats(d, score):
    """Point statistics on one (possibly resampled) ID data set."""
    appr = d["v_same"].values == 1
    wrong = d["correct"].values == 0
    w = d["w"].values
    lad = A.ladder(d)
    lad_det = A.ladder(d, use_e3=False)
    n = len(d)
    ex_b1 = appr & wrong
    r_b1 = (w * ex_b1).sum() / n
    r_b7 = (w * (lad_det["exec"].values & wrong)).sum() / n
    r_b8 = (w * (appr & lad_det["exec"].values & wrong)).sum() / n
    # fixed budget: escalate BUDGET share of approved actions to the full ladder
    prev = w * (appr & wrong) * (1 - lad["exec"].values)       # loss removed if escalated
    na = appr.sum(); m = int(round(BUDGET * na))
    idx = np.where(appr)[0]

    def top(sc):
        o = idx[np.argsort(-sc[idx], kind="stable")][:m]
        return r_b1 - prev[o].sum() / n
    r_cave = top(score)
    r_conf = top(1 - d["c_same"].values)
    r_rand = r_b1 - (m / na) * prev[idx].sum() / n              # expectation of random escalation
    y = wrong[appr].astype(int)
    if 0 < y.sum() < len(y):
        auc_c = roc_auc_score(y, score[appr]); auc_f = roc_auc_score(y, 1 - d["c_same"].values[appr])
    else:
        auc_c = auc_f = np.nan
    return {"B1": r_b1, "B7": r_b7, "B8": r_b8, "CAVE20": r_cave, "conf20": r_conf, "rand20": r_rand,
            "auc_cave": auc_c, "auc_conf": auc_f}


def boot_tests(df, score):
    idf = df[df.group != "unfamiliar"].reset_index(drop=True)
    sc = score[(df.group != "unfamiliar").values]
    point = stats(idf, sc)
    cl = idf.cluster.values; uc, codes = np.unique(cl, return_inverse=True)
    rows_of = [np.where(codes == i)[0] for i in range(len(uc))]
    rng = np.random.default_rng(1)
    B = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uc), len(uc))
        ix = np.concatenate([rows_of[p] for p in pick])
        d = idf.iloc[ix].reset_index(drop=True); d.attrs = df.attrs
        B.append(stats(d, sc[ix]))
    B = pd.DataFrame(B)
    comps = {"B7 - B1": ("B7", "B1"), "B8 - B7": ("B8", "B7"), "CAVE - random (20%)": ("CAVE20", "rand20"),
             "CAVE - confidence (20%)": ("CAVE20", "conf20"), "AUROC CAVE - confidence": ("auc_cave", "auc_conf")}
    out = {}
    for name, (a, b) in comps.items():
        diff = (B[a] - B[b]).dropna()
        est = point[a] - point[b]
        p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        out[name] = {"est": float(est), "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
                     "p": float(min(1.0, max(p, 1 / N_BOOT)))}
    return point, out


def cluster_crc(df, score_fn_seed=2):
    """Task-level CRC: per-task loss = sum of action losses over all runs of that task; target alpha * n_bar."""
    rng = np.random.default_rng(score_fn_seed)
    shift = df[df.group == "unfamiliar"].copy(); shift.attrs = df.attrs
    idf = df[df.group != "unfamiliar"]
    ids = idf.cluster.unique()
    sf = lambda d: ((d["correct"] == 0) & (d["v_same"] == 1)).values * d["w"].values
    wsum = idf.groupby("cluster")["w"].sum()
    Bt = float(wsum.max())                                      # largest possible loss of any task
    res = []
    for s in range(N_SPLIT):
        p = rng.permutation(ids); a, b = int(.4 * len(p)), int(.7 * len(p))
        dev, cal, test = (df[df.cluster.isin(p[:a])], df[df.cluster.isin(p[a:b])], df[df.cluster.isin(p[b:])])
        for x in (dev, cal, test):
            x.attrs = df.attrs
        est = A.fit_estimator(dev, A.FEATS)
        r_cal, r_te = A.predict(est, cal), A.predict(est, test)
        la = sf(cal) * (1 - A.ladder(cal)["exec"].values)
        ccodes, cu = pd.factorize(cal.cluster.values)
        nbar = len(cal) / len(cu); nC = len(cu)
        row = {"split": s}
        for al in (0.005, 0.01, 0.02, 0.05):
            thr = -np.inf
            for t in np.r_[np.inf, np.unique(r_cal)[::-1]]:
                Lc = np.bincount(ccodes, np.where(r_cal >= t, 0.0, la), minlength=nC)
                if (nC / (nC + 1)) * Lc.mean() + Bt / (nC + 1) <= al * nbar:
                    thr = t; break
            for nm, d, r in (("id", test, r_te), ("shift", shift, A.predict(est, shift))):
                if not len(d):
                    continue
                L = np.where(r >= thr, 0.0, sf(d) * (1 - A.ladder(d)["exec"].values))
                tcodes, tu = pd.factorize(d.cluster.values)
                row[f"{nm}_{al}"] = float(L.sum() / len(d))
                row[f"{nm}_task_{al}"] = float(np.bincount(tcodes, L).mean() / nbar)
                row[f"{nm}_esc_{al}"] = float(((r >= thr) & (d["v_same"].values == 1)).mean())
        res.append(row)
    R = pd.DataFrame(res)
    summ = {}
    for col in R.columns:
        if col == "split":
            continue
        v = R[col].dropna()
        al = float(col.split("_")[-1])
        summ[col] = [float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
                     float((v <= al).mean()) if "esc" not in col else None]
    return summ, Bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ev", choices=["v1", "v2"])
    ev = ap.parse_args().ev
    res = {}
    for m in MODELS:
        df, _ = load_model(m, ev)
        score = crossfit(df)
        point, tests = boot_tests(df, score)
        crc, Bt = cluster_crc(df)
        res[m] = {"point": point, "tests": tests, "cluster_crc": crc, "B_task": Bt}
        print(m, "done", flush=True)
    # Holm over all tests except AUROC-direction-free ones: every comparison is a family member
    fam = [(m, t) for m in res for t in res[m]["tests"]]
    ps = sorted(fam, key=lambda x: res[x[0]]["tests"][x[1]]["p"])
    k = len(ps); run_max = 0
    for i, (m, t) in enumerate(ps):
        adj = min(1.0, (k - i) * res[m]["tests"][t]["p"]); run_max = max(run_max, adj)
        res[m]["tests"][t]["p_holm"] = run_max
    (OUT / "stats").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "stats" / f"stats_{ev}.json", "w"), indent=1)
    print("done", ev)


if __name__ == "__main__":
    main()
