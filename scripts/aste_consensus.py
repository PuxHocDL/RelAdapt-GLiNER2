"""Cross-seed prediction stability: is agreement a correctness signal?

Costs no training - it reuses the five seeds already on disk. For a domain, every
triplet predicted by at least one seed gets a consensus count

    c(y) = |{s : y in Yhat_s}|   in 1..5

and the derived prediction sets are `Yhat_>=k = {y : c(y) >= k}`, so k=1 is the
union across seeds and k=5 the intersection.

Why this and not another architecture arm. Data order alone moves dev F1 by ~0.7
in this project, R1bm's gain is largely selectivity, and S1 showed false
positives *are* identifiable but not removable within a recall budget. If true
positives are stable across training trajectories while false positives are not,
then disagreement is itself a selectivity signal that no single model currently
uses - and the follow-up is to distil `c(y)/5` into a single-model head rather
than to add more context or more loss terms.

Three tables:

1. P/R/F1 against k.
2. The consensus distribution split by TP and FP, with `E[c|TP]` vs `E[c|FP]`.
   A large gap is the whole hypothesis.
3. For every *gold* triplet, how many of the five seeds found it. `0/5` is a
   systematic representation failure; `2/5` or `3/5` means the model is capable
   and the training trajectory is what is unstable. These need different fixes.

Plus a component breakdown - aspect span, opinion span, pair, polarity, full
triplet - to locate where the instability lives.

    python scripts/aste_consensus.py --arm R1bm-cf --seeds 62,63,64,65,66
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
DOMAINS = ("14lap", "14res", "15res", "16res")


def load(domain: str, arm: str, seeds: set[int], split: str):
    key = "dev_predictions" if split == "dev" else "predictions"
    per_seed = {}
    for p in sorted(glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json"))):
        x = json.loads(Path(p).read_text(encoding="utf-8"))
        s = int(x["seed"])
        if seeds and s not in seeds:
            continue
        per_seed[s] = {
            r["id"]: ({(tuple(a), tuple(o), q) for a, o, q in r["gold"]},
                      {(tuple(a), tuple(o), q) for a, o, q in r["pred"]})
            for r in x.get(key, [])
        }
    return per_seed


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R1bm-cf")
    ap.add_argument("--seeds", default="62,63,64,65,66")
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    args = ap.parse_args()
    seeds = {int(s) for s in args.seeds.split(",") if s}
    n_seed = len(seeds)

    data = {d: load(d, args.arm, seeds, args.split) for d in DOMAINS}
    data = {d: v for d, v in data.items() if len(v) == n_seed}
    if not data:
        print("no runs on disk for that arm/seeds")
        return 1
    print(f"# Cross-seed consensus, arm `{args.arm}`, {n_seed} seeds, {args.split}\n")

    # ---------- 1. P/R/F1 against k -----------------------------------------
    print("## 1. Prediction set at consensus >= k\n")
    print("| k | P | R | F1 | predictions kept |")
    print("|---" * 5 + "|")
    per_k = {}
    for k in range(1, n_seed + 1):
        Fs, Ps, Rs, kept = [], [], [], 0
        for d, per_seed in data.items():
            tp = npd = ngd = 0
            ids = sorted(next(iter(per_seed.values())))
            for sid in ids:
                gold = per_seed[min(per_seed)][sid][0]
                c = Counter()
                for s in per_seed:
                    c.update(per_seed[s][sid][1])
                keep = {y for y, n in c.items() if n >= k}
                tp += len(keep & gold)
                npd += len(keep)
                ngd += len(gold)
            p, r, f = prf(tp, npd, ngd)
            Ps.append(p); Rs.append(r); Fs.append(f); kept += npd
        per_k[k] = (statistics.mean(Ps), statistics.mean(Rs), statistics.mean(Fs))
        print(f"| >={k}/{n_seed} | {100 * per_k[k][0]:.2f} | {100 * per_k[k][1]:.2f} "
              f"| **{100 * per_k[k][2]:.2f}** | {kept} |")
    single = []
    for d, per_seed in data.items():
        fs = []
        for s in per_seed:
            tp = sum(len(per_seed[s][i][1] & per_seed[s][i][0]) for i in per_seed[s])
            npd = sum(len(per_seed[s][i][1]) for i in per_seed[s])
            ngd = sum(len(per_seed[s][i][0]) for i in per_seed[s])
            fs.append(prf(tp, npd, ngd)[2])
        single.append(statistics.mean(fs))
    print(f"\n  single-seed average F1 for reference: **{100 * statistics.mean(single):.2f}**")

    # ---------- 2. consensus distribution, TP vs FP --------------------------
    print("\n## 2. Consensus distribution, true vs false predictions\n")
    tpc, fpc = Counter(), Counter()
    for d, per_seed in data.items():
        ids = sorted(next(iter(per_seed.values())))
        for sid in ids:
            gold = per_seed[min(per_seed)][sid][0]
            c = Counter()
            for s in per_seed:
                c.update(per_seed[s][sid][1])
            for y, n in c.items():
                (tpc if y in gold else fpc)[n] += 1
    tot_t, tot_f = sum(tpc.values()), sum(fpc.values())
    print("| c(y) | correct (TP) | share | wrong (FP) | share |")
    print("|---" * 5 + "|")
    for k in range(1, n_seed + 1):
        print(f"| {k}/{n_seed} | {tpc[k]} | {100 * tpc[k] / max(tot_t, 1):.1f}% "
              f"| {fpc[k]} | {100 * fpc[k] / max(tot_f, 1):.1f}% |")
    e_tp = sum(k * v for k, v in tpc.items()) / max(tot_t, 1)
    e_fp = sum(k * v for k, v in fpc.items()) / max(tot_f, 1)
    print(f"\n  **E[c | TP] = {e_tp:.2f}**   vs   **E[c | FP] = {e_fp:.2f}**   "
          f"(gap {e_tp - e_fp:+.2f})")
    hi_t = sum(tpc[k] for k in (n_seed - 1, n_seed)) / max(tot_t, 1)
    hi_f = sum(fpc[k] for k in (n_seed - 1, n_seed)) / max(tot_f, 1)
    print(f"  share at c >= {n_seed - 1}: TP {100 * hi_t:.1f}% vs FP {100 * hi_f:.1f}%")

    # ---------- 3. gold triplets: how many seeds found them ------------------
    print("\n## 3. Gold triplets, by how many seeds found them\n")
    gc = Counter()
    for d, per_seed in data.items():
        ids = sorted(next(iter(per_seed.values())))
        for sid in ids:
            gold = per_seed[min(per_seed)][sid][0]
            for g in gold:
                gc[sum(1 for s in per_seed if g in per_seed[s][sid][1])] += 1
    tot_g = sum(gc.values())
    print("| seeds finding it | count | share | reading |")
    print("|---" * 4 + "|")
    read = {0: "never found - systematic failure", n_seed: "always found - solved"}
    for k in range(0, n_seed + 1):
        print(f"| {k}/{n_seed} | {gc[k]} | {100 * gc[k] / max(tot_g, 1):.1f}% "
              f"| {read.get(k, 'trajectory-dependent')} |")
    unstable = sum(gc[k] for k in range(1, n_seed))
    print(f"\n  **{100 * gc[0] / max(tot_g, 1):.1f}% never found** by any seed; "
          f"**{100 * unstable / max(tot_g, 1):.1f}% found by some but not all** - "
          f"that second group is what trajectory stability could recover.")

    # ---------- 4. where the instability lives -------------------------------
    print("\n## 4. Component consensus (mean over predicted triplets)\n")
    comp = defaultdict(lambda: defaultdict(list))
    for d, per_seed in data.items():
        ids = sorted(next(iter(per_seed.values())))
        for sid in ids:
            gold = per_seed[min(per_seed)][sid][0]
            preds = {s: per_seed[s][sid][1] for s in per_seed}
            seen = set()
            for s in preds:
                for y in preds[s]:
                    if y in seen:
                        continue
                    seen.add(y)
                    a, o, pol = y
                    lab = "TP" if y in gold else "FP"
                    comp[lab]["aspect"].append(sum(1 for q in preds if any(t[0] == a for t in preds[q])))
                    comp[lab]["opinion"].append(sum(1 for q in preds if any(t[1] == o for t in preds[q])))
                    comp[lab]["pair"].append(sum(1 for q in preds if any(t[:2] == (a, o) for t in preds[q])))
                    comp[lab]["triplet"].append(sum(1 for q in preds if y in preds[q]))
    print("| component | mean c, TP | mean c, FP | gap |")
    print("|---" * 4 + "|")
    for key in ("aspect", "opinion", "pair", "triplet"):
        t = statistics.mean(comp["TP"][key]) if comp["TP"][key] else 0
        f = statistics.mean(comp["FP"][key]) if comp["FP"][key] else 0
        print(f"| {key} | {t:.2f} | {f:.2f} | {t - f:+.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
