"""Encoder control table: same matching layers, different encoders, seeds 62-66.

Mean +- sample std over five runs per dataset; the average is taken per run first.
Also checks that every arm on a seed saw the same mini-batch order.

    python analysis/encoder_control.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("14lap", "14res", "15res", "16res")
SEEDS = range(62, 67)
ARMS = [
    ("RelAdapt / GLiNER2 (reference, original order)", "R1bm-cf"),
    ("RelAdapt / GLiNER2", "R1bm-enc-gl2"),
    ("RelAdapt / DeBERTa-v3-base", "R1bm-enc-dv3"),
    ("RelAdapt / BERT-base-uncased", "R1bm-enc-bert"),
    ("Concat. matcher / GLiNER2 (reference, original order)", "B2h-cf"),
    ("Concat. matcher / DeBERTa-v3-base", "B2h-enc-dv3"),
]


def load(arm):
    out = {}
    for d in DOMAINS:
        for s in SEEDS:
            p = ROOT / f"results/aste_{d}/{arm}/seed_{s}/metrics.json"
            if p.exists():
                out[(d, s)] = json.loads(p.read_text(encoding="utf-8"))
    return out


def main() -> None:
    runs = {arm: load(arm) for _, arm in ARMS}
    fmt = lambda v: f"{st.mean(v):6.2f} ± {st.stdev(v):4.2f}" if len(v) > 1 else (  # noqa: E731
        f"{v[0]:6.2f}" if v else "   n/a")
    print(f"{'arm':54s} | " + " | ".join(f"{d:>13s}" for d in DOMAINS)
          + " | test avg      | dev avg       | n")
    for name, arm in ARMS:
        r = runs[arm]
        cells = [fmt([100 * r[(d, s)]["test"]["f1"] for s in SEEDS if (d, s) in r])
                 for d in DOMAINS]
        full = [s for s in SEEDS if all((d, s) in r for d in DOMAINS)]
        t_avg = [st.mean(100 * r[(d, s)]["test"]["f1"] for d in DOMAINS) for s in full]
        d_avg = [st.mean(100 * r[(d, s)]["dev_f1"] for d in DOMAINS) for s in full]
        print(f"{name:54s} | " + " | ".join(cells) + f" | {fmt(t_avg)} | {fmt(d_avg)} | {len(r)}")

    # Data-order check across the rng-neutral arms.
    neutral = ["R1bm-enc-gl2", "R1bm-enc-dv3", "R1bm-enc-bert", "B2h-enc-dv3"]
    bad = []
    for d in DOMAINS:
        for s in SEEDS:
            hs = {a: runs[a][(d, s)]["batch_order_hash"] for a in neutral if (d, s) in runs[a]}
            if len(set(hs.values())) > 1:
                bad.append((d, s, hs))
    print("\nbatch-order hashes identical across rng-neutral arms:", "yes" if not bad else bad)


if __name__ == "__main__":
    main()
