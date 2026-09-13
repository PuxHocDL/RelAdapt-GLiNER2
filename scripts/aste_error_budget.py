"""An empirical error budget for ASTE, from stored predictions.

Three oracles, each applied to the same runs, so the next module is chosen from
a measurement rather than from intuition.

**boundary** - every predicted span that overlaps a gold span of its own role is
snapped onto that gold span; the result is deduped and rescored. Two sloppy
predictions that snap onto the same gold pair collapse into one, and a snapped
triplet still has to carry the right link and the right label to count.

**pair** - the predicted span sets are held exactly as they are and only the
*links among real spans* are repaired: every gold triplet whose two spans both
appear in the predicted sets is emitted, and every predicted triplet that the
link oracle has no business touching (because at least one of its spans is not a
gold span of its role) is kept untouched. So a spurious span stays a spurious
span and precision is not secretly repaired.

**polarity** - spans and links are untouched; a predicted triplet whose (aspect,
opinion) pair is gold has its label replaced by the gold label.

Two views are printed. The **independent** view applies each oracle alone, which
is what makes the three directly comparable. The **cumulative** view nests them
polarity -> +pair -> +boundary, so the increments form a budget that adds up and
no error is counted twice - a triplet with both a boundary error and a wrong
label is fixed once, by whichever oracle comes first.

    python scripts/aste_error_budget.py --arm B2h
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Set from --split; "predictions" is the test split, "dev_predictions" the dev one.
PRED_KEY = ["dev_predictions"]
DOMAINS = ("14lap", "14res", "15res", "16res")


def overlaps(a, b) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def f1_of(tp, n_pred, n_gold):
    pr = tp / n_pred if n_pred else 0.0
    rc = tp / n_gold if n_gold else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def best_overlap(span, pool):
    best, best_n = None, 0
    for g in pool:
        if not overlaps(span, g):
            continue
        n = min(span[1], g[1]) - max(span[0], g[0])
        if n > best_n or (n == best_n and best is not None and g[0] < best[0]):
            best, best_n = g, n
    return best


def sentences(domain, arm, seeds=None):
    for path in sorted(glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if seeds and int(payload["seed"]) not in seeds:
            continue
        yield int(payload["seed"]), [
            ([(tuple(a), tuple(o), p) for a, o, p in s["gold"]],
             [(tuple(a), tuple(o), p) for a, o, p in s["pred"]])
            for s in payload.get(PRED_KEY[0], [])
        ]


# ---------------------------------------------------------------------------
# the three repairs, each taking (gold, pred) -> repaired pred set
# ---------------------------------------------------------------------------
def fix_polarity(gold, pred):
    by_pair = {(a, o): p for a, o, p in gold}
    return {(a, o, by_pair.get((a, o), p)) for a, o, p in pred}


def fix_pair(gold, pred):
    """Repair links among spans the model actually produced. Spans untouched."""
    g_asp = {t[0] for t in gold}
    g_opi = {t[1] for t in gold}
    p_asp = {t[0] for t in pred}
    p_opi = {t[1] for t in pred}
    out = {(a, o, p) for a, o, p in gold if a in p_asp and o in p_opi}
    # Anything whose spans are not both real stays exactly as predicted: the link
    # oracle cannot be credited for removing a span error.
    out |= {(a, o, p) for a, o, p in pred if a not in g_asp or o not in g_opi}
    return out


def fix_boundary(gold, pred):
    g_asp = {t[0] for t in gold}
    g_opi = {t[1] for t in gold}
    out = set()
    for a, o, p in pred:
        na = a if a in g_asp else (best_overlap(a, g_asp) or a)
        no = o if o in g_opi else (best_overlap(o, g_opi) or o)
        out.add((na, no, p))
    return out


def score(domain, arm, repairs, seeds=None):
    """Mean corpus F1 over seeds after applying `repairs` in order."""
    per_seed = []
    for _, sents in sentences(domain, arm, seeds):
        tp = n_pred = n_gold = 0
        for gold, pred in sents:
            cur = set(pred)
            for fn in repairs:
                cur = fn(gold, cur)
            G = set(gold)
            tp += len(G & cur)
            n_pred += len(cur)
            n_gold += len(G)
        per_seed.append(f1_of(tp, n_pred, n_gold))
    return statistics.mean(per_seed) if per_seed else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B2h")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--split", default="dev", choices=("dev", "test"),
                    help="which split's stored predictions to analyse")
    ap.add_argument("--json", default="results/aste_errors/error_budget.json")
    args = ap.parse_args()
    PRED_KEY[0] = "predictions" if args.split == "test" else "dev_predictions"
    print(f"<!-- analysing the {args.split} split -->")
    seeds = {int(s) for s in args.seeds.split(",") if s} or None
    report = {"arm": args.arm, "seeds": sorted(seeds) if seeds else "all", "domains": {}}

    print(f"# Error budget for {args.arm}"
          + (f", seeds {sorted(seeds)}" if seeds else "") + "\n")

    print("## Independent oracles (each applied alone)\n")
    print("| domain | base | +polarity | +pair | +boundary | d_pol | d_pair | d_bound |")
    print("|---" * 8 + "|")
    agg = {k: [] for k in ("base", "pol", "pair", "bound")}
    for d in DOMAINS:
        base = score(d, args.arm, [], seeds)
        pol = score(d, args.arm, [fix_polarity], seeds)
        pair = score(d, args.arm, [fix_pair], seeds)
        bound = score(d, args.arm, [fix_boundary], seeds)
        if base == 0.0:
            continue
        for k, v in (("base", base), ("pol", pol), ("pair", pair), ("bound", bound)):
            agg[k].append(v)
        report["domains"][d] = {"base": base, "polarity": pol, "pair": pair,
                                "boundary": bound}
        print(f"| {d} | {100 * base:.2f} | {100 * pol:.2f} | {100 * pair:.2f} "
              f"| {100 * bound:.2f} | {100 * (pol - base):+.2f} "
              f"| {100 * (pair - base):+.2f} | {100 * (bound - base):+.2f} |")
    m = {k: statistics.mean(v) for k, v in agg.items() if v}
    print(f"| **avg** | **{100 * m['base']:.2f}** | {100 * m['pol']:.2f} "
          f"| {100 * m['pair']:.2f} | {100 * m['bound']:.2f} "
          f"| **{100 * (m['pol'] - m['base']):+.2f}** "
          f"| **{100 * (m['pair'] - m['base']):+.2f}** "
          f"| **{100 * (m['bound'] - m['base']):+.2f}** |")

    order = sorted((("boundary", m["bound"] - m["base"]),
                    ("pair", m["pair"] - m["base"]),
                    ("polarity", m["pol"] - m["base"])), key=lambda kv: -kv[1])
    print("\nRanking: " + " > ".join(f"d_{k} ({100 * v:+.2f})" for k, v in order))
    report["independent_ranking"] = [(k, v) for k, v in order]

    print("\n## Cumulative (nested, so the increments add up)\n")
    print("| domain | base | +pol | +pol+pair | +pol+pair+bound "
          "| d_pol | d_pair | d_bound | unreachable |")
    print("|---" * 9 + "|")
    cum = {k: [] for k in ("base", "a", "b", "c")}
    for d in DOMAINS:
        base = score(d, args.arm, [], seeds)
        if base == 0.0:
            continue
        a = score(d, args.arm, [fix_polarity], seeds)
        b = score(d, args.arm, [fix_polarity, fix_pair], seeds)
        c = score(d, args.arm, [fix_polarity, fix_pair, fix_boundary], seeds)
        for k, v in (("base", base), ("a", a), ("b", b), ("c", c)):
            cum[k].append(v)
        report["domains"][d]["cumulative"] = {"polarity": a, "pair": b, "boundary": c}
        print(f"| {d} | {100 * base:.2f} | {100 * a:.2f} | {100 * b:.2f} | {100 * c:.2f} "
              f"| {100 * (a - base):+.2f} | {100 * (b - a):+.2f} | {100 * (c - b):+.2f} "
              f"| {100 * (1 - c):.2f} |")
    mc = {k: statistics.mean(v) for k, v in cum.items() if v}
    print(f"| **avg** | **{100 * mc['base']:.2f}** | {100 * mc['a']:.2f} "
          f"| {100 * mc['b']:.2f} | {100 * mc['c']:.2f} "
          f"| **{100 * (mc['a'] - mc['base']):+.2f}** "
          f"| **{100 * (mc['b'] - mc['a']):+.2f}** "
          f"| **{100 * (mc['c'] - mc['b']):+.2f}** | {100 * (1 - mc['c']):.2f} |")
    print("\n`unreachable` is what no oracle here can fix: a gold span that appears "
          "in no predicted triplet at all, even partially.")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
