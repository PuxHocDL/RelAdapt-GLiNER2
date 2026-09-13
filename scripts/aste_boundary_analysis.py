"""Does H0b do the thing it claims, and how much was there to do?

Two questions, kept apart on purpose.

**Mechanism.** If the listwise span objective works, boundary errors should fall
while the sentiment decision stays where it was. Anything else - a gain that
shows up as better polarity, or as more candidates surviving - means the arm won
for a reason other than the one it was built on, and the +0.41 average is then a
coincidence that will not generalise.

**Headroom.** The boundary oracle snaps every predicted span that overlaps a gold
span of its own role onto that gold span, dedupes, and recomputes ordinary exact
triplet F1. Unlike relaxing the match to overlap, this is a real decode the model
could in principle produce: two sloppy predictions that snap onto the same gold
pair collapse into one, and a snapped triplet still has to get the polarity and
the pairing right to count. The gap it opens is what a perfect boundary module
would be worth, and

    recovery ratio = (F1_H0b - F1_B2h) / (F1_oracle - F1_B2h)

says how much of that H0b actually took. A domain where H0b loses is only
damning if that domain had headroom to begin with.

    python scripts/aste_boundary_analysis.py
    python scripts/aste_boundary_analysis.py --arms B2h,H0b --json results/...
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Set from --split; "predictions" is the test split, "dev_predictions" the dev one.
PRED_KEY = ["dev_predictions"]
DOMAINS = ("14lap", "14res", "15res", "16res")


def overlaps(a, b) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def f1_of(tp: int, n_pred: int, n_gold: int) -> float:
    pr = tp / n_pred if n_pred else 0.0
    rc = tp / n_gold if n_gold else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def runs(domain: str, arm: str):
    """seed -> [(gold_triplets, pred_triplets), ...] in sentence order."""
    out = {}
    for path in sorted(glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        out[seed] = [
            ([(tuple(a), tuple(o), p) for a, o, p in s["gold"]],
             [(tuple(a), tuple(o), p) for a, o, p in s["pred"]],
             s["id"])
            for s in payload.get(PRED_KEY[0], [])
        ]
    return out


# ---------------------------------------------------------------------------
# span-level status
# ---------------------------------------------------------------------------
def status(span, pool):
    """exact / partial / absent, against the predicted spans of the same role."""
    if span in pool:
        return "exact"
    return "partial" if any(overlaps(span, x) for x in pool) else "absent"


def best_overlap(span, pool):
    """The gold span this prediction most plausibly meant, or None."""
    best, best_n = None, 0
    for g in pool:
        if not overlaps(span, g):
            continue
        n = min(span[1], g[1]) - max(span[0], g[0])
        if n > best_n or (n == best_n and best is not None and g[0] < best[0]):
            best, best_n = g, n
    return best


def side(pred, gold) -> str:
    """Which end of a partially-correct span is wrong."""
    left = pred[0] != gold[0]
    right = pred[1] != gold[1]
    if left and right:
        return "both"
    return "left" if left else ("right" if right else "exact")


def mechanism(domain: str, arm: str) -> dict:
    per_seed = defaultdict(list)
    for seed, sents in runs(domain, arm).items():
        c = Counter()
        for gold, pred, _ in sents:
            for role, gi, pi in (("aspect", 0, 0), ("opinion", 1, 1)):
                g_spans = {t[gi] for t in gold}
                p_spans = {t[pi] for t in pred}
                for g in g_spans:
                    st = status(g, p_spans)
                    width = "multi" if g[1] - g[0] > 1 else "single"
                    c[f"{role}_{st}"] += 1
                    c[f"{role}_total"] += 1
                    c[f"{width}_{st}"] += 1
                    c[f"{width}_total"] += 1
                    if st == "partial":
                        p = best_overlap(g, p_spans)
                        if p is not None:
                            c[f"side_{side(p, g)}"] += 1
                            c["side_total"] += 1
            # downstream, conditional on both spans being exactly available
            p_asp = {t[0] for t in pred}
            p_opi = {t[1] for t in pred}
            pred_set = set(pred)
            pred_pairs = {(a, o) for a, o, _ in pred}
            for a, o, lab in gold:
                if a not in p_asp or o not in p_opi:
                    continue
                c["cond_total"] += 1
                if (a, o, lab) in pred_set:
                    c["cond_hit"] += 1
                elif (a, o) in pred_pairs:
                    c["cond_wrong_polarity"] += 1
                else:
                    c["cond_not_paired"] += 1
        for k, v in c.items():
            per_seed[k].append(v)
    return {k: sum(v) for k, v in per_seed.items()}


# ---------------------------------------------------------------------------
# boundary oracle
# ---------------------------------------------------------------------------
def snap_oracle(domain: str, arm: str) -> dict:
    """Per-seed exact triplet F1 after snapping overlapping spans onto gold."""
    base, oracle = [], []
    moved = Counter()
    for seed, sents in runs(domain, arm).items():
        b = Counter()
        o = Counter()
        for gold, pred, _ in sents:
            G = set(gold)
            b["tp"] += len(G & set(pred))
            b["n_pred"] += len(set(pred))
            b["n_gold"] += len(G)

            g_asp = {t[0] for t in gold}
            g_opi = {t[1] for t in gold}
            snapped = set()
            for a, op, lab in pred:
                na = a if a in g_asp else (best_overlap(a, g_asp) or a)
                no = op if op in g_opi else (best_overlap(op, g_opi) or op)
                if na != a or no != op:
                    moved["snapped"] += 1
                snapped.add((na, no, lab))
            # Snapping can collapse two sloppy predictions onto one gold pair;
            # the set does that for us, and n_pred shrinks accordingly.
            moved["collapsed"] += len(pred) - len(snapped) if len(pred) > len(snapped) else 0
            o["tp"] += len(G & snapped)
            o["n_pred"] += len(snapped)
            o["n_gold"] += len(G)
        base.append(f1_of(b["tp"], b["n_pred"], b["n_gold"]))
        oracle.append(f1_of(o["tp"], o["n_pred"], o["n_gold"]))
    return {"f1": statistics.mean(base) if base else 0.0,
            "oracle_f1": statistics.mean(oracle) if oracle else 0.0,
            "n_seeds": len(base), **moved}


# ---------------------------------------------------------------------------
# cross-arm transition: does H0b convert partial into exact?
# ---------------------------------------------------------------------------
def transition(domain: str, ctrl: str, arm: str) -> Counter:
    a_runs, b_runs = runs(domain, ctrl), runs(domain, arm)
    shared = sorted(set(a_runs) & set(b_runs))
    c = Counter()
    for seed in shared:
        by_id = {sid: (g, p) for g, p, sid in b_runs[seed]}
        for gold, pred_a, sid in a_runs[seed]:
            if sid not in by_id:
                continue
            _, pred_b = by_id[sid]
            for role, i in (("aspect", 0), ("opinion", 1)):
                g_spans = {t[i] for t in gold}
                pa = {t[i] for t in pred_a}
                pb = {t[i] for t in pred_b}
                for g in g_spans:
                    c[f"{status(g, pa)}->{status(g, pb)}"] += 1
    c["n_seeds"] = len(shared)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B2h,H0b")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--split", default="dev", choices=("dev", "test"),
                    help="which split's stored predictions to analyse")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    PRED_KEY[0] = "predictions" if args.split == "test" else "dev_predictions"
    print(f"<!-- analysing the {args.split} split -->")
    arms = [a for a in args.arms.split(",") if a]
    doms = [d for d in args.domains.split(",") if d]
    ctrl, treat = arms[0], arms[-1]
    report = {}

    print(f"# boundary analysis: {treat} against {ctrl}\n")

    print("## 1. boundary oracle headroom and how much of it was taken\n")
    print("Snap every overlapping predicted span onto its gold span, dedupe, "
          "recompute exact triplet F1.\n")
    print(f"| domain | {ctrl} | snap oracle | headroom | {treat} | gain | recovery |")
    print("|---" * 7 + "|")
    tot = defaultdict(float)
    for d in doms:
        o = snap_oracle(d, ctrl)
        t = snap_oracle(d, treat)
        head = o["oracle_f1"] - o["f1"]
        gain = t["f1"] - o["f1"]
        ratio = gain / head if head > 1e-9 else float("nan")
        report.setdefault(d, {})["oracle"] = {
            "control_f1": o["f1"], "oracle_f1": o["oracle_f1"], "headroom": head,
            "treat_f1": t["f1"], "gain": gain, "recovery_ratio": ratio,
            "spans_snapped": o.get("snapped", 0)}
        tot["ctrl"] += o["f1"]; tot["orc"] += o["oracle_f1"]; tot["tr"] += t["f1"]
        rr = "n/a" if ratio != ratio else f"{100 * ratio:+.0f}%"
        print(f"| {d} | {100 * o['f1']:.2f} | {100 * o['oracle_f1']:.2f} | "
              f"+{100 * head:.2f} | {100 * t['f1']:.2f} | {100 * gain:+.2f} | {rr} |")
    n = len(doms)
    head = (tot["orc"] - tot["ctrl"]) / n
    gain = (tot["tr"] - tot["ctrl"]) / n
    print(f"| **avg** | {100 * tot['ctrl'] / n:.2f} | {100 * tot['orc'] / n:.2f} | "
          f"+{100 * head:.2f} | {100 * tot['tr'] / n:.2f} | {100 * gain:+.2f} | "
          f"{100 * gain / head:+.0f}% |")

    print("\n## 2. where the spans go wrong (pooled seeds, % of gold spans)\n")
    print("| domain | arm | aspect exact | opinion exact | partial | absent "
          "| single-tok exact | multi-tok exact |")
    print("|---" * 8 + "|")
    for d in doms:
        for a in arms:
            m = mechanism(d, a)
            report.setdefault(d, {}).setdefault("mechanism", {})[a] = m
            at, ot = max(m.get("aspect_total", 1), 1), max(m.get("opinion_total", 1), 1)
            gt = at + ot
            part = m.get("aspect_partial", 0) + m.get("opinion_partial", 0)
            absent = m.get("aspect_absent", 0) + m.get("opinion_absent", 0)
            st, mt = max(m.get("single_total", 1), 1), max(m.get("multi_total", 1), 1)
            print(f"| {d} | {a} | {100 * m.get('aspect_exact', 0) / at:.1f} "
                  f"| {100 * m.get('opinion_exact', 0) / ot:.1f} "
                  f"| {100 * part / gt:.1f} | {100 * absent / gt:.1f} "
                  f"| {100 * m.get('single_exact', 0) / st:.1f} "
                  f"| {100 * m.get('multi_exact', 0) / mt:.1f} |")

    print("\n## 3. which end is wrong, and what happens once the spans are right\n")
    print("| domain | arm | left | right | both | polarity acc given spans "
          "| paired given spans |")
    print("|---" * 7 + "|")
    for d in doms:
        for a in arms:
            m = report[d]["mechanism"][a]
            s = max(m.get("side_total", 1), 1)
            ct = max(m.get("cond_total", 1), 1)
            paired = m.get("cond_hit", 0) + m.get("cond_wrong_polarity", 0)
            print(f"| {d} | {a} | {100 * m.get('side_left', 0) / s:.1f} "
                  f"| {100 * m.get('side_right', 0) / s:.1f} "
                  f"| {100 * m.get('side_both', 0) / s:.1f} "
                  f"| {100 * m.get('cond_hit', 0) / max(paired, 1):.1f} "
                  f"| {100 * paired / ct:.1f} |")

    print(f"\n## 4. per-gold-span transition, {ctrl} -> {treat} (same seeds)\n")
    print("| domain | partial->exact | exact->partial | absent->exact "
          "| exact->absent | net exact |")
    print("|---" * 6 + "|")
    for d in doms:
        c = transition(d, ctrl, treat)
        report.setdefault(d, {})["transition"] = dict(c)
        gained = c["partial->exact"] + c["absent->exact"]
        lost = c["exact->partial"] + c["exact->absent"]
        print(f"| {d} | {c['partial->exact']} | {c['exact->partial']} "
              f"| {c['absent->exact']} | {c['exact->absent']} | **{gained - lost:+d}** |")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
