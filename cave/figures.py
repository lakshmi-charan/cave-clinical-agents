"""Paper figures (static PNG, 300 dpi) from results_all.json."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import out_dir

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e6e5e0", "#9b9a95"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})


def _short(a):
    return a


def fig_sf_by_condition(res, path):
    ags = [a for a in res if a != "cost_ledger"]
    fig, axes = plt.subplots(1, len(ags), figsize=(3.4 * len(ags), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    conds = [("base", "Base\ntasks"), ("unfamiliar", "Unfamiliar\ntools"), ("hazard", "Clinical\nhazards"),
             ("tier2", "Tier 2\nwrites")]
    pairs = [("same model", S1), ("same family", S2), ("cross family", S3)]
    for ax, ag in zip(axes, ags):
        tab = res[ag]["descriptive"]["pairs"]
        x = np.arange(len(conds))
        wdt = 0.26
        for i, (p, col) in enumerate(pairs):
            vals, lo, hi = [], [], []
            for c, _ in conds:
                v = tab[p].get(f"ESFR_{c}", [np.nan] * 3)
                vals.append(100 * v[0]); lo.append(100 * (v[0] - v[1])); hi.append(100 * (v[2] - v[0]))
            ax.bar(x + (i - 1) * (wdt + 0.02), vals, wdt, color=col, label=p, yerr=[lo, hi],
                   error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2})
        ax.set_xticks(x, [c[1] for c in conds], rotation=0, fontsize=7.5)
        ax.set_title(f"Agent: {ag}", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Executed shared failures\n(% of actions)")
    fig.legend(*axes[0].get_legend_handles_labels(), title="Verifier", fontsize=7.5, title_fontsize=7.5,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_risk_coverage(res, path):
    ags = [a for a in res if a != "cost_ledger"]
    fig, axes = plt.subplots(1, len(ags), figsize=(3.4 * len(ags), 2.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, ag in zip(axes, ags):
        cv = res[ag]["summary"].get("curves")
        if not cv:
            continue
        f = 100 * np.asarray(cv["frac"])
        for key, col, lab in [("CAVE", S1, "CAVE risk score"), ("confidence", S2, "Verifier confidence"),
                              ("random", GRAY, "Random")]:
            ax.plot(f, 100 * np.asarray(cv[key]), color=col, lw=2, label=lab,
                    ls="--" if key == "random" else "-")
        ax.set_xlabel("Approved actions escalated (%)")
        ax.set_title(f"Agent: {ag}", fontsize=9)
    axes[0].set_ylabel("Risk-weighted executed\nshared failures (x100)")
    axes[-1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_guarantee(res, path):
    ags = [a for a in res if a != "cost_ledger"]
    fig, axes = plt.subplots(1, len(ags), figsize=(3.4 * len(ags), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ag in zip(axes, ags):
        g = res[ag]["summary"]["guarantee"]
        alphas = sorted({float(k.split("_")[-1]) for k in g if k.startswith("id_0")})
        for pre, col, lab in [("id", S1, "In-distribution"), ("shift", S2, "Unfamiliar tools, unweighted"),
                              ("shiftw", S3, "Unfamiliar tools, weighted")]:
            ys = [g.get(f"{pre}_{a}", [np.nan] * 4) for a in alphas]
            if all(np.isnan(y[0]) for y in ys):
                continue
            m = [100 * y[0] for y in ys]
            ax.plot([100 * a for a in alphas], m, marker="o", ms=5, color=col, lw=2, label=lab)
            ax.fill_between([100 * a for a in alphas], [100 * y[1] for y in ys], [100 * y[2] for y in ys], color=col,
                            alpha=0.12, lw=0)
        lim = 100 * max(alphas) * 1.15
        ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls=":", label="Target (risk = alpha)")
        ax.set_xlabel("Target risk level alpha (x100)")
        ax.set_title(f"Agent: {ag}", fontsize=9)
    axes[0].set_ylabel("Realized risk on test (x100)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_tradeoff(res, path):
    ags = [a for a in res if a != "cost_ledger"]
    fig, axes = plt.subplots(1, len(ags), figsize=(3.4 * len(ags), 3.0))
    axes = np.atleast_1d(axes)
    for ax, ag in zip(axes, ags):
        pol = res[ag]["summary"]["id"]
        for name, m in pol.items():
            x, y = 1000 * m["cost_per_task"][0], 100 * m["wrisk"][0]
            is_c = name == "CAVE"
            ax.scatter([x], [y], s=46 if is_c else 26, color=S1 if is_c else GRAY, zorder=3,
                       edgecolor="white", linewidth=1)
            lab = {"B1 same model": "B1-same", "B1 same family": "B1-fam", "B1 cross family": "B1-cross",
                   "B2 panel k=3": "B2-k3", "B2 panel k=5": "B2-k5"}.get(name, name.split(" ")[0])
            ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=INK if is_c else INK2)
        ax.set_xlabel("Verification cost per task (USD x 1000)")
        ax.set_title(f"Agent: {ag}", fontsize=9)
    axes[0].set_ylabel("Risk-weighted executed\nshared failures (x100)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_coef(res, path):
    ags = [a for a in res if a != "cost_ledger"]
    fig, axes = plt.subplots(1, len(ags), figsize=(3.4 * len(ags), 2.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, ag in zip(axes, ags):
        c = res[ag]["summary"].get("coef") or {}
        if not c:
            continue
        items = sorted(c.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color=S1, height=0.6)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"Agent: {ag}", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Standardized logistic coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_all(res):
    d = out_dir("results", "figures")
    fig_sf_by_condition(res, d / "fig4_sf_by_condition.png")
    fig_risk_coverage(res, d / "fig5_risk_coverage.png")
    fig_guarantee(res, d / "fig6_guarantee.png")
    fig_tradeoff(res, d / "fig7_tradeoff.png")
    fig_coef(res, d / "fig8_coefficients.png")
