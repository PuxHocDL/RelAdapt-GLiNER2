"""Every number of the backbone version of the paper (main backbone DeBERTa-v3-base).

Reads stored predictions only. Writes analysis/backbone_paper.json and the stability
figure figures/stability-analysis-dv3.pdf.

    python analysis/backbone_paper.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import paper_numbers as pn  # noqa: E402
import paper_stats as ps  # noqa: E402
from sinkhorn_degree import multi_recall  # noqa: E402

DOMAINS = ("14lap", "14res", "15res", "16res")
S62, S67, S72, S10 = set(range(62, 67)), set(range(67, 72)), set(range(72, 77)), set(range(67, 77))
SFX = "-enc-dv3"
ENCODERS = (("BERT-base", "-enc-bert"), ("RoBERTa-base", "-enc-rob"),
            ("DeBERTa-v3-base", "-enc-dv3"), ("DeBERTa-v3-large", "-enc-dv3l"))
OUT = HERE / "backbone_paper.json"


def payload(d, arm, s):
    return json.loads((ROOT / f"results/aste_{d}/{arm}/seed_{s}/metrics.json").read_text(encoding="utf-8"))


def arm_summary(arm, seeds):
    """Per-domain mean +- std of P/R/F1 (test), and per-run averages for dev/test F1."""
    seeds = sorted(seeds)
    out = {"domains": {}}
    for d in DOMAINS:
        xs = [payload(d, arm, s) for s in seeds]
        out["domains"][d] = {k: [100 * x["test"][k] for x in xs] for k in ("precision", "recall", "f1")}
        out["domains"][d]["dev_f1"] = [100 * x["dev_f1"] for x in xs]
    for key, src in (("test_avg", "f1"), ("dev_avg", "dev_f1")):
        per_run = [st.mean(out["domains"][d][src][i] for d in DOMAINS) for i in range(len(seeds))]
        out[key] = [st.mean(per_run), st.stdev(per_run)]
    for d in DOMAINS:
        out["domains"][d] = {k: [st.mean(v), st.stdev(v)] for k, v in out["domains"][d].items()}
    return out


def consensus_prf(arm, seeds, split, k):
    data = [pn.load(d, arm, seeds, split) for d in DOMAINS]
    return {d: [100 * v for v in pn.consensus(x, k)] for d, x in zip(DOMAINS, data)}


def member_prf(arm, seeds, split):
    data = [pn.load(d, arm, seeds, split) for d in DOMAINS]
    return {d: [100 * v for v in pn.single(x)[:3]] for d, x in zip(DOMAINS, data)}


def gold_taxonomy(arm, seeds, split):
    found = Counter()
    for d in DOMAINS:
        rows = {s: ps.sets(payload(d, arm, s), split) for s in sorted(seeds)}
        first = rows[min(seeds)]
        for i in range(len(first)):
            for g in first[i][0]:
                found[sum(1 for s in seeds if g in rows[s][i][1])] += 1
    n = sum(found.values())
    K = len(seeds)
    return {"solved": 100 * found[K] / n, "dependent": 100 * sum(found[c] for c in range(1, K)) / n,
            "miss": 100 * found[0] / n, "n_gold": n}


def selectivity(ctrl, treat, seeds, split="test"):
    fp_removed = fp_added = gold_gained = gold_lost = 0
    for d in DOMAINS:
        for s in sorted(seeds):
            a, b = ps.sets(payload(d, ctrl, s), split), ps.sets(payload(d, treat, s), split)
            for (g, pa), (_, pb) in zip(a, b):
                fa, fb = pa - g, pb - g
                fp_removed += len(fa - fb); fp_added += len(fb - fa)
                gold_gained += len((pb & g) - pa); gold_lost += len((pa & g) - pb)
    return dict(fp_removed=fp_removed, fp_added=fp_added, gold_gained=gold_gained, gold_lost=gold_lost)


def tuned_threshold(arm, seeds):
    df, dr = [], []
    for d in DOMAINS:
        for s in sorted(seeds):
            x = payload(d, arm, s)
            df.append(100 * (x["test_tuned"]["f1"] - x["test"]["f1"]))
            dr.append(100 * (x["test_tuned"]["recall"] - x["test"]["recall"]))
    return {"delta_f1": st.mean(df), "delta_recall": st.mean(dr)}


def neu_counts(arm, seeds, domain="15res", split="test"):
    pred = gold = tp = 0
    for s in sorted(seeds):
        for g, y in ps.sets(payload(domain, arm, s), split):
            pred += sum(1 for t in y if t[2] == 2)
            gold += sum(1 for t in g if t[2] == 2)
            tp += sum(1 for t in y & g if t[2] == 2)
    return {"neu_pred": pred, "neu_gold": gold, "neu_tp": tp}


def error_budget(arm, seeds, split="dev"):
    tmp = HERE / f"_error_budget_{arm}_{split}.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/aste_error_budget.py"), "--arm", arm,
                    "--seeds", ",".join(map(str, sorted(seeds))), "--split", split,
                    "--json", str(tmp)], check=True, capture_output=True,
                   env={**os.environ, "PYTHONUTF8": "1"})
    rep = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink()
    return rep


def main() -> None:
    res = {}
    arms = ("A0", "B2", "B2h", "R1m", "R1b", "R1bm")
    res["dv3_arms"] = {a: arm_summary(a + SFX, S62) for a in arms}
    res["dv3_groups"] = {name: arm_summary("R1bm" + SFX, s) for name, s in
                         (("A", S62), ("B", S67), ("C", S72), ("BC", S10))}
    res["backbones"] = {}
    for name, sfx in ENCODERS:
        res["backbones"][name] = {a: arm_summary(a + sfx, S62) for a in ("B2h", "R1bm")}
        for split in ("dev", "test"):
            res["backbones"][name][f"boot_{split}"] = ps.boot_arm_diff("B2h" + sfx, "R1bm" + sfx, S62, split)
        print(name, {a: res["backbones"][name][a]["test_avg"] for a in ("B2h", "R1bm")}, flush=True)

    r1 = "R1bm" + SFX
    res["prf"] = {"concat": member_prf("B2h" + SFX, S62, "test"), "reladapt": member_prf(r1, S62, "test"),
                  "cons5_A": consensus_prf(r1, S62, "test", 3), "cons10": consensus_prf(r1, S10, "test", 6)}
    res["consensus"] = {}
    for split in ("dev", "test"):
        for name, s, k in (("A", S62, 3), ("B", S67, 3), ("C", S72, 3), ("BC", S10, 6)):
            res["consensus"][f"{name}_{split}"] = {"prf": consensus_prf(r1, s, split, k),
                                                   "members": member_prf(r1, s, split),
                                                   "boot": ps.boot_consensus(r1, s, split, k)}
        res[f"boot_reladapt_vs_concat_{split}"] = ps.boot_arm_diff("B2h" + SFX, r1, S62, split)
        res[f"size_curve_{split}"] = ps.size_curve(r1, S10, split)
        res[f"k_sweep_{split}"] = {k: consensus_prf(r1, S10, split, k) for k in range(1, 11)}
    res["funnel"] = {a: ps.funnel(a + SFX, S62) for a in ("B2h", "R1bm")}
    res["gold_taxonomy"] = {f"{n}_{sp}": gold_taxonomy(r1, s, sp)
                            for n, s in (("A", S62), ("B", S67), ("C", S72)) for sp in ("dev", "test")}
    res["selectivity_test"] = selectivity("B2h" + SFX, r1, S62)
    res["tuned_threshold_B"] = tuned_threshold(r1, S67)
    res["neu_15res"] = {a: neu_counts(a + SFX, S62) for a in ("B2h", "R1bm")}
    res["shared_span_recall"] = {}
    for a in ("A0", "B2", "B2h", "R1bm"):
        r, share = multi_recall(str(ROOT / ("results/aste_{d}/" + a + SFX)), S62, "predictions")
        res["shared_span_recall"][a] = {**r, "share": share}
    res["error_budget_dev"] = {a: error_budget(a + SFX, S62) for a in ("B2h", "R1bm")}
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT, flush=True)

    # Stability figure, same panels as the GLiNER2 version, on the DeBERTa-v3 groups.
    curve = {f"size_curve_{sp}": res[f"size_curve_{sp}"] for sp in ("dev", "test")}
    (HERE / "_curve_dv3.json").write_text(json.dumps(curve), encoding="utf-8")
    r = subprocess.run([sys.executable, str(HERE / "make_figures.py")], capture_output=True, text=True,
                       env={**os.environ, "FIG_ARM_A": r1, "FIG_ARM_BC": r1, "PYTHONUTF8": "1",
                            "FIG_STATS": "_curve_dv3.json", "FIG_OUT": "stability-analysis-dv3.pdf"})
    print(r.stdout, r.stderr[-2000:])
    (HERE / "_curve_dv3.json").unlink()


if __name__ == "__main__":
    main()
