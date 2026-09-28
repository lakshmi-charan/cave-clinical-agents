"""Per-run summary table and the result figures of the paper (Figs. 4-8).

Run after analysis/pooled.py (both evidence versions, tag "main").
output: runs/analysis/per_run.csv, runs/analysis/policies_<EV>.csv and runs/analysis/figures/*.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import HELD_OUT, LABEL, MODELS, OUT, load_run  # noqa: E402

BLUE, ORANGE, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.frameon": False})
FIG = OUT / "figures"
TITLE = {"gpt-5-mini": "gpt-5-mini (3 runs)", "gpt-5": "GPT-5 (2 runs)", "claude-haiku-4-5": "Claude Haiku 4.5 (1 run)"}
POLICIES = ["B0 no verifier", "B1 same model", "B1 same family", "B1 cross family", "B2 panel k=3", "B2 panel k=5",
            "B3 low-confidence escalation", "B4 rules on all writes", "B5 verify everything", "B6 random escalation",
            "B7 checks only", "B8 verifier + checks", "CAVE"]


def contradicts(d, v):
    return (d[f"E1{v}"] == "contradict") | (d[f"E2{v}"] == "contradict")


def per_run():
    rows = []
    for model, runs in MODELS.items():
        for i, r in enumerate(runs):
            U, T = load_run(r)
            w, c = U[U.correct == 0], U[U.correct == 1]
            base = T[T.stressor == "base"]
            leak = ~(contradicts(w, "v2"))
            rows.append(dict(
                run=r, label=f"{LABEL[model]} r{i + 1}", held_out=r in HELD_OUT, actions=len(U), wrong=len(w),
                error_rate=len(w) / len(U),
                catch_E1E2=((w.E1 == "contradict") | (w.E2 == "contradict")).mean(), catch_E1E2_plus=contradicts(w, "v2").mean(),
                false_alarm_E1E2_plus=contradicts(c, "v2").mean(),
                base_success_strict=base.success_strict.mean(), base_success_clinical=base.success.mean(),
                self_approves_wrong=w.v_same.mean(), self_rejects_correct=(c.v_same == 0).mean(),
                missed_by_E1E2_plus=int(leak.sum()),
                missed_omissions=int((leak & w.why.str.contains("missing", na=False)).sum())))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "per_run.csv", index=False)
    return D


def style(ax, title=None):
    if title:
        ax.set_title(title, color=INK, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load_pooled(ev="v2"):
    P = {}
    for m in MODELS:
        R = json.load(open(OUT / "pooled" / m / f"{ev}_main.json"))
        G = R["summary"]["guarantee"]
        P[m] = {"R": R, "pairs": R["descriptive"]["pairs"], "curves": R["summary"]["curves"], "guar": G,
                "alphas": sorted({float(k.split("_")[-1]) for k in G if k.startswith("id_0")})}
    return P


def policy_table(ev):
    rows = []
    for m in MODELS:
        S = json.load(open(OUT / "pooled" / m / f"{ev}_main.json"))["summary"]["id"]
        for p in POLICIES:
            if p in S:
                v = S[p]
                rows.append({"model": m, "policy": p, "risk_x100": 100 * v["wrisk"][0],
                             "escalated_pct": 100 * v["esc_rate"][0], "correct_blocked_pct": 100 * v["blocked_correct"][0],
                             "clinician_min_per_100_tasks": v["e4_min_per_100_tasks"][0]})
    pd.DataFrame(rows).round(3).to_csv(OUT / f"policies_{ev}.csv", index=False)


def fig_evidence(D):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x, w = np.arange(len(D)), 0.38
    ax.bar(x - w / 2, 100 * D.catch_E1E2, w * 0.94, color=ORANGE, label="E1/E2 (pre-specified)")
    ax.bar(x + w / 2, 100 * D.catch_E1E2_plus, w * 0.94, color=BLUE, label="E1+/E2+ (strengthened)")
    ax.set_xticks(x, [l + ("*" if h else "") for l, h in zip(D.label, D.held_out)], fontsize=7.5)
    ax.set_ylabel("Wrong actions refuted (%)")
    ax.set_ylim(0, 100)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "evidence_coverage.png", dpi=300)
    plt.close(fig)


def fig_shared_failures(P):
    conds = [("base", "Base"), ("unfamiliar", "Unfam.\ntools"), ("hazard", "Hazards"), ("tier2", "Medic.\norders")]
    series = [("same model", BLUE, "Same model"), ("same family", ORANGE, "Same family"),
              ("cross family", GREEN, "Cross family"), ("panel k=5", AMBER, "Panel k = 5")]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.35), sharey=True)
    for ax, m in zip(axs, MODELS):
        pairs = P[m]["pairs"]
        ss = [s for s in series if s[0] in pairs]
        x, w = np.arange(len(conds)), 0.8 / len(ss)
        for i, (k, col, lab) in enumerate(ss):
            v = np.array([pairs[k].get(f"ESFR_{c}", [np.nan] * 3) for c, _ in conds]) * 100
            ax.bar(x + (i - (len(ss) - 1) / 2) * w, v[:, 0], w * 0.9, color=col, label=lab,
                   yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], error_kw={"elinewidth": 0.6, "ecolor": INK2, "capsize": 1.5})
        ax.set_xticks(x, [c[1] for c in conds])
        style(ax, TITLE[m])
    axs[0].set_ylabel("Executed shared failures (%)")
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "shared_failures.png", dpi=300)
    plt.close(fig)


def fig_risk_coverage(P):
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2))
    for ax, m in zip(axs, MODELS):
        cv = P[m]["curves"]
        f = 100 * np.asarray(cv["frac"])
        for k, col, lab, ls in [("CAVE", BLUE, "CAVE risk score", "-"), ("confidence", ORANGE, "Verifier confidence", "-"),
                                ("random", GRAY, "Random", "--")]:
            ax.plot(f, 100 * np.asarray(cv[k]), color=col, lw=1.6, ls=ls, label=lab)
        ax.set_xlabel("Approved actions escalated (%)")
        ax.set_ylim(bottom=0)
        style(ax, TITLE[m])
    axs[0].set_ylabel("Risk (× 100)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "risk_coverage.png", dpi=300)
    plt.close(fig)


def fig_tradeoff(P):
    short = {"CAVE": "CAVE", "B0 no verifier": "B0", "B1 same model": "B1s", "B5 verify everything": "B5",
             "B7 checks only": "B7"}
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for ax, m in zip(axs, MODELS):
        pts = [(n, v["e4_min_per_100_tasks"][0], 100 * v["wrisk"][0]) for n, v in P[m]["R"]["summary"]["id"].items()]
        xmax = max(p[1] for p in pts) or 1
        for n, xv, yv in pts:
            col = BLUE if n == "CAVE" else (GREEN if n in ("B7 checks only", "B8 verifier + checks") else GRAY)
            ax.scatter([xv], [yv], s=26 if n == "CAVE" else 14, color=col, zorder=3, edgecolor="white", lw=0.6)
            if n in short:
                ax.annotate(short[n], (xv, yv), xytext=(3, -8 if n == "B7 checks only" else 3), textcoords="offset points",
                            fontsize=6.8, color=INK if n == "CAVE" else INK2, weight="bold" if n == "CAVE" else "normal")
        ax.set_xlim(-0.05 * xmax, 1.15 * xmax)
        ax.set_xlabel("Clinician min / 100 tasks")
        style(ax, TITLE[m])
    axs[0].set_ylabel("Risk (× 100)")
    fig.tight_layout()
    fig.savefig(FIG / "risk_vs_clinician_time.png", dpi=300)
    plt.close(fig)


def fig_guarantee(P):
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    for ax, m in zip(axs, MODELS):
        G, al = P[m]["guar"], P[m]["alphas"]
        for pre, col, lab in [("id", BLUE, "In-distribution"), ("shift", ORANGE, "Tool shift, unweighted"),
                              ("shiftw", GREEN, "Tool shift, weighted")]:
            ys = np.array([G[f"{pre}_{a}"][:3] for a in al]) * 100
            ax.plot([100 * a for a in al], ys[:, 0], marker="o", ms=3, color=col, lw=1.5, label=lab)
            ax.fill_between([100 * a for a in al], ys[:, 1], ys[:, 2], color=col, alpha=0.12, lw=0)
        lim = 100 * max(al) * 1.05
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls=":", label="Risk = α")
        ax.set_xlabel("Target α (× 100)")
        style(ax, TITLE[m])
    axs[0].set_ylabel("Realized excess risk (× 100)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "guarantee_validity.png", dpi=300)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    D = per_run()
    print(D.round(3).to_string(index=False))
    fig_evidence(D)
    for ev in ("v2", "v1"):
        policy_table(ev)
    P = load_pooled("v2")
    fig_shared_failures(P)
    fig_risk_coverage(P)
    fig_tradeoff(P)
    fig_guarantee(P)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
