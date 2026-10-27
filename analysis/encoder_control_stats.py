"""Paired sentence bootstrap for the encoder-control arms (matched mini-batch order).

Writes analysis/encoder_control_stats.json.

    python analysis/encoder_control_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import paper_stats as ps

SEEDS = set(range(62, 67))
COMPARISONS = [
    ("DeBERTa-v3 vs GLiNER2 (RelAdapt)", "R1bm-enc-gl2", "R1bm-enc-dv3"),
    ("GLiNER2 vs BERT (RelAdapt)", "R1bm-enc-bert", "R1bm-enc-gl2"),
    ("biaffine gain on DeBERTa-v3", "B2h-enc-dv3", "R1bm-enc-dv3"),
]


def main() -> None:
    out = {}
    for name, ctrl, treat in COMPARISONS:
        for split in ("dev", "test"):
            r = ps.boot_arm_diff(ctrl, treat, SEEDS, split)
            out[f"{name} | {split}"] = r
            per = ", ".join(f"{d} {v:+.2f}" for d, v in r["per_domain"].items())
            print(f"{name:34s} {split:4s} diff {r['diff']:+.2f} "
                  f"CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p={r['p_one_sided']:.4f} | {per}")
    Path(__file__).with_name("encoder_control_stats.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
