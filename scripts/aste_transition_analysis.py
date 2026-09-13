"""Where did R1bm's +1.22 actually come from?

The existence metrics said the biaffine arm rejects *more* gold relations as NONE
(deep rejection 13.7% -> 14.2%, emit recall 83.5% -> 82.3%) while gaining F1 and
precision on all four domains. So the funnel story does not explain the win, and
the design of the next arm should not be based on it.

This decomposes the gain by matching the two arms sentence-by-sentence and
seed-by-seed and asking what changed about each gold triplet and each false
positive. Two very different explanations are separable here:

  * mostly ``wrong polarity / wrong partner / boundary -> hit`` means the
    biaffine improved **relational discrimination** - it picks the right partner
    and label among competing candidates;
  * mostly ``false positive -> gone`` means it improved **selectivity** - it
    emits less junk without understanding relations any better.

Both raise F1. Only the first argues for richer pair-conditioned context.

    python scripts/aste_transition_analysis.py --a B2h-c2 --b R1bm-c2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("14lap", "14res", "15res", "16res")


def overlaps(x, y) -> bool:
    return not (x[1] <= y[0] or y[1] <= x[0])


def load(domain: str, arm: str):
    out = {}
    for p in sorted(glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json"))):
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        out[int(payload["seed"])] = {
            r["id"]: ([(tuple(a), tuple(o), q) for a, o, q in r["gold"]],
                      [(tuple(a), tuple(o), q) for a, o, q in r["pred"]])
            for r in payload.get("dev_predictions", [])
        }
    return out


def gold_fate(t, pred) -> str:
    """What happened to one gold triplet under a prediction set."""
    a, o, lab = t
    P = set(pred)
    if (a, o, lab) in P:
        return "hit"
    pairs = {(x, y) for x, y, _ in pred}
    if (a, o) in pairs:
        return "wrong_polarity"
    p_asp = {x for x, _, _ in pred}
    p_opi = {y for _, y, _ in pred}
    if a in p_asp or o in p_opi:
        return "wrong_partner"          # one span emitted, bound to the wrong one
    if any(overlaps(a, x) for x in p_asp) or any(overlaps(o, y) for y in p_opi):
        return "boundary"
    return "missing"


def fp_type(t, gold) -> str:
    a, o, lab = t
    g_asp = {x for x, _, _ in gold}
    g_opi = {y for _, y, _ in gold}
    pairs = {(x, y) for x, y, _ in gold}
    if (a, o) in pairs:
        return "fp_wrong_polarity"
    if a in g_asp and o in g_opi:
        return "fp_wrong_partner"
    if a in g_asp or o in g_opi:
        return "fp_one_span_real"
    if any(overlaps(a, x) for x in g_asp) or any(overlaps(o, y) for y in g_opi):
        return "fp_boundary"
    return "fp_spurious"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="B2h-c2", help="baseline arm")
    ap.add_argument("--b", default="R1bm-c2", help="new arm")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args()
    doms = [d for d in args.domains.split(",") if d]

    print(f"# Transition {args.a} -> {args.b}, dev, matched by seed and sentence\n")
    grand = Counter()
    for d in doms:
        A, B = load(d, args.a), load(d, args.b)
        shared = sorted(set(A) & set(B))
        c = Counter()
        for seed in shared:
            for sid, (gold, pred_a) in A[seed].items():
                if sid not in B[seed]:
                    continue
                _, pred_b = B[seed][sid]
                for t in gold:
                    fa, fb = gold_fate(t, pred_a), gold_fate(t, pred_b)
                    c[f"{fa}->{fb}"] += 1
                ga = set(gold)
                fa_set = {t for t in pred_a if t not in ga}
                fb_set = {t for t in pred_b if t not in ga}
                for t in fa_set - fb_set:
                    c["FPremoved_" + fp_type(t, gold)] += 1
                for t in fb_set - fa_set:
                    c["FPadded_" + fp_type(t, gold)] += 1
        grand.update(c)
        report(d, c, len(shared))
    print("\n## pooled over all domains\n")
    report("ALL", grand, None)
    return 0


def report(name, c, n_seeds):
    gains = {k: v for k, v in c.items() if k.endswith("->hit") and not k.startswith("hit")}
    losses = {k: v for k, v in c.items() if k.startswith("hit->") and k != "hit->hit"}
    n_gain, n_loss = sum(gains.values()), sum(losses.values())
    removed = {k: v for k, v in c.items() if k.startswith("FPremoved_")}
    added = {k: v for k, v in c.items() if k.startswith("FPadded_")}

    head = f"## {name}" + (f"  ({n_seeds} seeds)" if n_seeds else "")
    print(f"\n{head}\n")
    print(f"**Recall side** - gold triplets newly hit: **+{n_gain}**, lost: **-{n_loss}**, "
          f"net **{n_gain - n_loss:+d}**")
    for k, v in sorted(gains.items(), key=lambda kv: -kv[1]):
        print(f"    {k:32} {v:5d}")
    for k, v in sorted(losses.items(), key=lambda kv: -kv[1]):
        print(f"    {k:32} {v:5d}")
    tot_r, tot_a = sum(removed.values()), sum(added.values())
    print(f"\n**Precision side** - false positives removed: **{tot_r}**, "
          f"added: **{tot_a}**, net **{tot_a - tot_r:+d}**")
    for k, v in sorted(removed.items(), key=lambda kv: -kv[1]):
        print(f"    {k:32} -{v}")
    for k, v in sorted(added.items(), key=lambda kv: -kv[1]):
        print(f"    {k:32} +{v}")
    if tot_r + n_gain:
        share = 100 * n_gain / (n_gain + tot_r)
        print(f"\n  of the improvement, {share:.0f}% is relational "
              f"(a gold triplet becomes correct) and {100 - share:.0f}% is "
              f"selectivity (a false positive disappears)")


if __name__ == "__main__":
    sys.exit(main())
