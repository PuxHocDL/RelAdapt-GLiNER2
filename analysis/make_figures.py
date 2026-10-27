"""Figures for the RelAdapt-GLiNER2 paper, computed from stored predictions.

    python analysis/paper_stats.py      # writes analysis/paper_stats.json (size curve, CIs)
    python analysis/make_figures.py     # writes figures/stability-analysis.pdf

Panel (a) and (b) recompute consensus statistics from results/aste_*/; panel (c)
reads the ensemble-size curve from paper_stats.json.
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_stats import DOMAINS, payloads, sets

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "figures"
BLUE, ORANGE, GREY = "#1f5fa8", "#d9822b", "#7a7a7a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def consensus_stats(arm, seeds, split):
    """Consensus distribution by TP/FP and mean component consensus."""
    K = len(seeds)
    tp, fp = Counter(), Counter()
    comp = {lab: {c: [] for c in ("aspect", "opinion", "pair", "triplet")} for lab in ("TP", "FP")}
    for d in DOMAINS:
        P = payloads(d, arm, seeds)
        rows = {s: sets(P[s], split) for s in sorted(seeds)}
        first = rows[min(seeds)]
        for i in range(len(first)):
            gold = first[i][0]
            preds = [rows[s][i][1] for s in sorted(seeds)]
            seen = set()
            for ps in preds:
                for y in ps:
                    if y in seen:
                        continue
                    seen.add(y)
                    a, o, _ = y
                    lab = "TP" if y in gold else "FP"
                    c = sum(1 for q in preds if y in q)
                    (tp if lab == "TP" else fp)[c] += 1
                    comp[lab]["aspect"].append(sum(1 for q in preds if any(t[0] == a for t in q)) / K)
                    comp[lab]["opinion"].append(sum(1 for q in preds if any(t[1] == o for t in q)) / K)
                    comp[lab]["pair"].append(sum(1 for q in preds if any(t[:2] == (a, o) for t in q)) / K)
                    comp[lab]["triplet"].append(c / K)
    means = {lab: {c: st.mean(v) for c, v in comp[lab].items()} for lab in comp}
    return tp, fp, means


def main() -> None:
    import os

    # Defaults reproduce the GLiNER2 figure; the backbone paper overrides them.
    arm_a = os.environ.get("FIG_ARM_A", "R1bm-cf")
    arm_bc = os.environ.get("FIG_ARM_BC", "R1bm-cs")
    stats = HERE / os.environ.get("FIG_STATS", "paper_stats.json")
    IMAGES.mkdir(exist_ok=True)
    primary = (arm_a, set(range(62, 67)), "dev")
    replications = [(arm_bc, set(range(67, 72)), "dev"), (arm_bc, set(range(72, 77)), "dev"),
                    (arm_bc, set(range(67, 72)), "test"), (arm_bc, set(range(72, 77)), "test")]
    tp, fp, means = consensus_stats(*primary)
    reps = [consensus_stats(*r)[2] for r in replications]
    curve = json.loads(stats.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))

    # (a) consensus distribution
    ax = axes[0]
    ks = [1, 2, 3, 4, 5]
    nt, nf = sum(tp.values()), sum(fp.values())
    w = 0.38
    ax.bar([k - w / 2 for k in ks], [100 * tp[k] / nt for k in ks], w, color=BLUE, label="correct (TP)")
    ax.bar([k + w / 2 for k in ks], [100 * fp[k] / nf for k in ks], w, color=ORANGE, label="wrong (FP)")
    e_tp = sum(k * v for k, v in tp.items()) / nt
    e_fp = sum(k * v for k, v in fp.items()) / nf
    ax.set_xticks(ks, [f"{k}/5" for k in ks])
    ax.set_xlabel("consensus count $c(y)$")
    ax.set_ylabel("share of predictions (%)")
    ax.text(0.03, 0.95, f"$E[c\\,|\\,\\mathrm{{TP}}]={e_tp:.2f}$\n$E[c\\,|\\,\\mathrm{{FP}}]={e_fp:.2f}$",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.set_ylim(0, 84)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80))
    ax.set_title("(a) Triplet consensus, TP vs. FP")

    # (b) component stability gap
    ax = axes[1]
    comps = ["aspect", "opinion", "pair", "triplet"]
    gaps = [5 * (means["TP"][c] - means["FP"][c]) for c in comps]
    ax.bar(range(4), gaps, 0.6, color=[GREY, GREY, BLUE, BLUE], alpha=0.85)
    for j, r in enumerate(reps):
        ax.scatter([i + (j - 1.5) * 0.09 for i in range(4)],
                   [5 * (r["TP"][c] - r["FP"][c]) for c in comps],
                   s=7, color="black", zorder=3, label="replications" if j == 0 else None)
    for i, g in enumerate(gaps):
        top = max([g] + [5 * (r["TP"][comps[i]] - r["FP"][comps[i]]) for r in reps])
        ax.text(i, top + 0.08, f"+{g:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(4), ["Aspect", "Opinion", "Pair", "Triplet"])
    ax.set_ylabel("TP $-$ FP mean consensus (of 5)")
    ax.set_ylim(0, 2.45)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Stability gap by component")

    # (c) ensemble size
    ax = axes[2]
    for split, color, name in (("test", BLUE, "test"), ("dev", ORANGE, "dev")):
        c = curve[f"size_curve_{split}"]
        K = sorted(int(k) for k in c)
        f = [c[str(k)]["f1"] for k in K]
        sd = [c[str(k)]["f1_sd"] for k in K]
        ax.plot(K, f, marker="o", ms=2.5, lw=1.0, color=color, label=name)
        ax.fill_between(K, [a - b for a, b in zip(f, sd)], [a + b for a, b in zip(f, sd)],
                        color=color, alpha=0.18, lw=0)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("trajectories $K$ (inference cost $K\\times$)")
    ax.set_ylabel("triplet F1 (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(c) Consensus at $\\tau=0.6$")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    out = IMAGES / os.environ.get("FIG_OUT", "stability-analysis.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    print("E[c|TP], E[c|FP] =", round(e_tp, 2), round(e_fp, 2))
    print("gaps", dict(zip(comps, [round(g, 2) for g in gaps])))
    for r, spec in zip(reps, replications):
        print(spec[1] and (min(spec[1]), spec[2]), {c: round(5 * (r["TP"][c] - r["FP"][c]), 2) for c in comps})


if __name__ == "__main__":
    main()
