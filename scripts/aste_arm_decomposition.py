"""Per-arm decomposition: localisation, linking, sentiment, or calibration?

Final ASTE F1 cannot distinguish an arm that localises spans better from one
that merely shifts which candidates survive. This prints, for every arm against
a control on the same seeds:

  exact ASTE P/R/F1
  aspect  exact-span P/R/F1        opinion exact-span P/R/F1
  partial-overlap error rate       exact<->partial transitions vs the control
  link accuracy | predicted exact spans
  sentiment accuracy | correct pair

One item on the request list is **not** computable from stored predictions and is
reported as missing rather than approximated: *link accuracy conditional on gold
spans*. That needs a teacher-forced pass with the gold spans injected into the
candidate set, and the training script does not retain weights, so it would mean
re-running. `--teacher-forced` says whether a run carries it; runs made before
that diagnostic existed will say `n/a`.

    python scripts/aste_arm_decomposition.py --arms B2h,J1,J2 --seeds 47,48,49,50,51
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Set from --split; "predictions" is the test split, "dev_predictions" the dev one.
PRED_KEY = ["dev_predictions"]
DOMAINS = ("14lap", "14res", "15res", "16res")


def overlaps(a, b) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def prf(tp, n_pred, n_gold):
    pr = tp / n_pred if n_pred else 0.0
    rc = tp / n_gold if n_gold else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def load(domain, arm, seeds):
    out = {}
    for path in sorted(glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json"))):
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        s = int(p["seed"])
        if seeds and s not in seeds:
            continue
        # Runs made before dev predictions were stored have none for --split dev.
        rows = p.get(PRED_KEY[0]) or []
        if not rows:
            continue
        out[s] = {r["id"]: ([(tuple(a), tuple(o), q) for a, o, q in r["gold"]],
                            [(tuple(a), tuple(o), q) for a, o, q in r["pred"]])
                  for r in rows}
    return out


def stats(runs) -> dict:
    c = Counter()
    for seed, sents in runs.items():
        for gold, pred in sents.values():
            G, P = set(gold), set(pred)
            c["tp"] += len(G & P)
            c["n_pred"] += len(P)
            c["n_gold"] += len(G)
            for role, i in (("asp", 0), ("opi", 1)):
                g = {t[i] for t in gold}
                q = {t[i] for t in pred}
                c[f"{role}_tp"] += len(g & q)
                c[f"{role}_npred"] += len(q)
                c[f"{role}_ngold"] += len(g)
                for x in g:
                    if x in q:
                        continue
                    c["partial" if any(overlaps(x, y) for y in q) else "absent"] += 1
                c["span_gold"] += len(g)
            # linking and sentiment, conditional on the exact spans being present
            p_asp = {t[0] for t in pred}
            p_opi = {t[1] for t in pred}
            pairs = {(a, o) for a, o, _ in pred}
            exact = set(pred)
            for a, o, lab in gold:
                if a not in p_asp or o not in p_opi:
                    continue
                c["cond_n"] += 1
                if (a, o) in pairs:
                    c["cond_linked"] += 1
                    c["cond_right_label"] += int((a, o, lab) in exact)
    return dict(c)


def transitions(ctrl_runs, arm_runs) -> Counter:
    c = Counter()
    for seed in sorted(set(ctrl_runs) & set(arm_runs)):
        a, b = ctrl_runs[seed], arm_runs[seed]
        for sid, (gold, pred_a) in a.items():
            if sid not in b:
                continue
            _, pred_b = b[sid]
            for i in (0, 1):
                g = {t[i] for t in gold}
                qa = {t[i] for t in pred_a}
                qb = {t[i] for t in pred_b}
                for x in g:
                    sa = "exact" if x in qa else ("partial" if any(overlaps(x, y) for y in qa) else "absent")
                    sb = "exact" if x in qb else ("partial" if any(overlaps(x, y) for y in qb) else "absent")
                    c[f"{sa}->{sb}"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B2h,J1,J2")
    ap.add_argument("--control", default="B2h")
    ap.add_argument("--seeds", default="47,48,49,50,51")
    ap.add_argument("--split", default="dev", choices=("dev", "test"),
                    help="which split's stored predictions to analyse")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    PRED_KEY[0] = "predictions" if args.split == "test" else "dev_predictions"
    print(f"<!-- analysing the {args.split} split -->")
    arms = [a for a in args.arms.split(",") if a]
    seeds = {int(s) for s in args.seeds.split(",") if s}
    report = {}

    print(f"# Arm decomposition, seeds {sorted(seeds)}\n")
    print("## Triplet and span quality\n")
    print("| domain | arm | n | ASTE P | ASTE R | ASTE F1 | asp-span F1 | opi-span F1 "
          "| partial % | absent % | link acc | sent acc |")
    print("|---" * 12 + "|")
    means = {a: [] for a in arms}
    for d in DOMAINS:
        for a in arms:
            runs = load(d, a, seeds)
            if not runs:
                continue
            s = stats(runs)
            report.setdefault(d, {})[a] = s
            p, r, f = prf(s["tp"], s["n_pred"], s["n_gold"])
            _, _, af = prf(s["asp_tp"], s["asp_npred"], s["asp_ngold"])
            _, _, of = prf(s["opi_tp"], s["opi_npred"], s["opi_ngold"])
            gs = max(s.get("span_gold", 1), 1)
            cn = max(s.get("cond_n", 1), 1)
            cl = max(s.get("cond_linked", 1), 1)
            means[a].append(f)
            print(f"| {d} | {a} | {len(runs)} | {100 * p:.2f} | {100 * r:.2f} "
                  f"| **{100 * f:.2f}** | {100 * af:.2f} | {100 * of:.2f} "
                  f"| {100 * s.get('partial', 0) / gs:.2f} | {100 * s.get('absent', 0) / gs:.2f} "
                  f"| {100 * s.get('cond_linked', 0) / cn:.2f} "
                  f"| {100 * s.get('cond_right_label', 0) / cl:.2f} |")
        print("|" + "---|" * 12)
    print("\n## Pooled average triplet F1\n")
    ctrl_mean = statistics.mean(means[args.control]) if means.get(args.control) else 0.0
    for a in arms:
        if not means[a]:
            continue
        mm = statistics.mean(means[a])
        tag = "" if a == args.control else f"  ({100 * (mm - ctrl_mean):+.2f} vs {args.control})"
        print(f"- **{a}**: {100 * mm:.2f}{tag}")

    print(f"\n## Gold-span transitions vs {args.control} (same seeds and sentences)\n")
    print("| domain | arm | partial->exact | exact->partial | absent->exact "
          "| exact->absent | net exact |")
    print("|---" * 7 + "|")
    for d in DOMAINS:
        ctrl = load(d, args.control, seeds)
        for a in arms:
            if a == args.control:
                continue
            runs = load(d, a, seeds)
            if not runs or not ctrl:
                continue
            t = transitions(ctrl, runs)
            gained = t["partial->exact"] + t["absent->exact"]
            lost = t["exact->partial"] + t["exact->absent"]
            report.setdefault(d, {}).setdefault("transitions", {})[a] = dict(t)
            print(f"| {d} | {a} | {t['partial->exact']} | {t['exact->partial']} "
                  f"| {t['absent->exact']} | {t['exact->absent']} | **{gained - lost:+d}** |")

    print("\n> `link acc` = share of gold triplets whose two spans both appear "
          "exactly among the predictions that are also linked to each other.\n"
          "> `sent acc` = share of those correctly-linked pairs carrying the right "
          "label.\n"
          "> Link accuracy *conditional on gold spans* is **not** reported: it needs "
          "a teacher-forced pass and the trainer does not retain weights.")

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
