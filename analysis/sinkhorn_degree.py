"""Does the Sinkhorn assignment prior hurt many-to-many relations?

1. Gold relation degree: share of aspect (opinion) spans linked to more than one
   opinion (aspect) within a sentence, over all splits of ASTE-Data-V2.
2. Recall on gold triplets whose aspect or opinion takes part in another triplet
   ("multi") versus one-to-one triplets, for A0 (no prior) vs B2 (prior), which
   differ only in the assignment prior, and for RelAdapt-GLiNER2.

    python analysis/sinkhorn_degree.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from aste_data import load_split  # noqa: E402

DOMAINS = ("14lap", "14res", "15res", "16res")


def degrees():
    a_deg, o_deg = Counter(), Counter()
    for d in DOMAINS:
        for split in ("train", "dev", "test"):
            for ex in load_split(d, split, str(ROOT / "Data/ASTE")):
                trips = {(tuple(t[0]), tuple(t[1])) for t in _triplets(ex)}
                ca = Counter(a for a, _ in trips)
                co = Counter(o for _, o in trips)
                a_deg.update(ca.values())
                o_deg.update(co.values())
    return a_deg, o_deg


def _triplets(ex):
    return [(t.aspect, t.opinion) for t in ex.triplets]


def multi_recall(root_fmt, seeds, key):
    """Mean over seeds and domains of recall on multi vs one-to-one gold triplets."""
    rec = {"multi": [], "single": []}
    share = []
    for d in DOMAINS:
        paths = [p for p in glob.glob(root_fmt.format(d=d) + "/seed_*/metrics.json")]
        per = {"multi": [], "single": []}
        for p in paths:
            x = json.loads(Path(p).read_text(encoding="utf-8"))
            if int(x["seed"]) not in seeds:
                continue
            hit = Counter(); tot = Counter()
            for r in x[key]:
                gold = [(tuple(a), tuple(o), q) for a, o, q in r["gold"]]
                pred = {(tuple(a), tuple(o), q) for a, o, q in r["pred"]}
                ca = Counter(a for a, _, _ in set(gold))
                co = Counter(o for _, o, _ in set(gold))
                for g in set(gold):
                    kind = "multi" if ca[g[0]] > 1 or co[g[1]] > 1 else "single"
                    tot[kind] += 1
                    hit[kind] += g in pred
            for k in per:
                per[k].append(hit[k] / tot[k])
            share.append(tot["multi"] / (tot["multi"] + tot["single"]))
        assert len(per["multi"]) == len(seeds), (root_fmt, d, len(per["multi"]))
        for k in rec:
            rec[k].append(st.mean(per[k]))
    return {k: 100 * st.mean(v) for k, v in rec.items()}, 100 * st.mean(share)


def main():
    a_deg, o_deg = degrees()
    na, no = sum(a_deg.values()), sum(o_deg.values())
    print(f"aspects with degree > 1: {100 * sum(v for k, v in a_deg.items() if k > 1) / na:.1f}% "
          f"(max {max(a_deg)}, n={na})")
    print(f"opinions with degree > 1: {100 * sum(v for k, v in o_deg.items() if k > 1) / no:.1f}% "
          f"(max {max(o_deg)}, n={no})")
    S42, S62, S67 = set(range(42, 47)), set(range(62, 67)), set(range(67, 77))
    for name, fmt, seeds, key in (
        ("A0 no prior, test", str(ROOT / "results/aste_{d}/A0"), S42, "predictions"),
        ("B2 prior, test", str(ROOT / "results/aste_{d}/B2"), S42, "predictions"),
        ("B2h-cf, dev", str(ROOT / "results/aste_{d}/B2h-cf"), S62, "dev_predictions"),
        ("R1bm-cf, dev", str(ROOT / "results/aste_{d}/R1bm-cf"), S62, "dev_predictions"),
        ("R1bm-cs, test", str(ROOT / "results/aste_{d}/R1bm-cs"), S67, "predictions"),
    ):
        r, share = multi_recall(fmt, seeds, key)
        print(f"{name:18s} recall multi {r['multi']:.2f} | one-to-one {r['single']:.2f} "
              f"| gap {r['multi'] - r['single']:+.2f} | multi share {share:.1f}%")


if __name__ == "__main__":
    main()
