"""The pre-specified confirmatory test for H0b vs B2h on the fresh seeds.

Primary: mean exact triplet F1 pooled over the four domains, paired bootstrap
over test sentences, 10,000 resamples. Secondary: per-domain, Holm-corrected.
See ``results/aste_h0b_confirm/PREREGISTRATION.md`` - written before these runs
existed and not edited since.

The bootstrap resamples *sentences within a domain* and recomputes corpus P/R/F1
from the resampled tp / n_pred / n_gold, per seed, exactly as ``aste_stats.py``
does. The two arms are resampled with the same sentence indices, so the pairing
is preserved; domains are then averaged with equal weight, which is what the
registered primary metric says.

    python scripts/aste_confirm_test.py --seeds 47,48,49,50,51
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("14lap", "14res", "15res", "16res")
N_BOOT = 10000


def counts(domain: str, arm: str, seeds: set[int]):
    """seed -> per-sentence (tp, n_pred, n_gold) arrays, sentence order fixed."""
    out = {}
    for path in sorted(glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        if seed not in seeds:
            continue
        rows = sorted(payload["predictions"], key=lambda s: s["id"])
        out[seed] = np.array([[s["tp"], s["n_pred"], s["n_gold"]] for s in rows], dtype=float)
    return out


def corpus_f1(arr: np.ndarray) -> float:
    tp, np_, ng = arr.sum(axis=0)
    pr = tp / np_ if np_ else 0.0
    rc = tp / ng if ng else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="B2h")
    ap.add_argument("--treat", default="H0b")
    ap.add_argument("--seeds", default="47,48,49,50,51")
    ap.add_argument("--json", default="results/aste_h0b_confirm/confirm.json")
    args = ap.parse_args()
    seeds = {int(s) for s in args.seeds.split(",") if s}
    rng = np.random.default_rng(20260912)

    data, report = {}, {"control": args.control, "treat": args.treat,
                        "seeds": sorted(seeds), "n_bootstrap": N_BOOT, "domains": {}}
    for d in DOMAINS:
        c, t = counts(d, args.control, seeds), counts(d, args.treat, seeds)
        shared = sorted(set(c) & set(t))
        if not shared:
            print(f"!! {d}: no paired runs on seeds {sorted(seeds)}")
            return 1
        if len(shared) != len(seeds):
            print(f"!! {d}: only {len(shared)} of {len(seeds)} seeds paired - "
                  f"the registered design needs all of them")
        data[d] = (c, t, shared)

    print(f"# Confirmatory: {args.treat} vs {args.control}, seeds {sorted(seeds)}\n")
    print("## Per-domain means (secondary)\n")
    print(f"| domain | n | {args.control} | {args.treat} | diff |")
    print("|---" * 5 + "|")
    obs_by_domain = {}
    for d in DOMAINS:
        c, t, shared = data[d]
        cf = statistics.mean(corpus_f1(c[s]) for s in shared)
        tf = statistics.mean(corpus_f1(t[s]) for s in shared)
        obs_by_domain[d] = (cf, tf)
        print(f"| {d} | {len(shared)} | {100 * cf:.2f} | {100 * tf:.2f} | {100 * (tf - cf):+.2f} |")
    ctrl_avg = statistics.mean(v[0] for v in obs_by_domain.values())
    treat_avg = statistics.mean(v[1] for v in obs_by_domain.values())
    observed = treat_avg - ctrl_avg
    print(f"| **avg** | | **{100 * ctrl_avg:.2f}** | **{100 * treat_avg:.2f}** "
          f"| **{100 * observed:+.2f}** |")

    # ---- primary: pooled paired bootstrap ----
    boot = np.empty(N_BOOT)
    per_domain_boot = {d: np.empty(N_BOOT) for d in DOMAINS}
    for b in range(N_BOOT):
        diffs = []
        for d in DOMAINS:
            c, t, shared = data[d]
            n = c[shared[0]].shape[0]
            idx = rng.integers(0, n, n)
            dc = statistics.mean(corpus_f1(c[s][idx]) for s in shared)
            dt = statistics.mean(corpus_f1(t[s][idx]) for s in shared)
            per_domain_boot[d][b] = dt - dc
            diffs.append(dt - dc)
        boot[b] = float(np.mean(diffs))

    lo, hi = np.percentile(boot, [2.5, 97.5])
    centred = boot - boot.mean()
    p = float((np.abs(centred) >= abs(observed)).mean())
    report["primary"] = {"observed_diff": observed, "ci95": [float(lo), float(hi)],
                         "p_value": p, "significant_0.05": bool(p < 0.05 and observed > 0)}

    print("\n## Primary (pre-specified): pooled average over the four domains\n")
    print("| quantity | value |\n|---|---|")
    print(f"| observed diff | **{100 * observed:+.3f}** F1 |")
    print(f"| bootstrap 95% CI | [{100 * lo:+.3f}, {100 * hi:+.3f}] |")
    print(f"| two-sided p | **{p:.4f}** |")
    verdict = ("CONFIRMED" if p < 0.05 and observed > 0 else "NOT CONFIRMED")
    print(f"| registered decision rule | **{verdict}** |")

    print("\n## Per-domain bootstrap (secondary, Holm across 4)\n")
    raw = []
    for d in DOMAINS:
        bd = per_domain_boot[d]
        o = obs_by_domain[d][1] - obs_by_domain[d][0]
        pv = float((np.abs(bd - bd.mean()) >= abs(o)).mean())
        raw.append((d, o, float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)), pv))
    order = sorted(range(4), key=lambda i: raw[i][4])
    holm = {}
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(running, raw[i][4] * (4 - rank)))
        running = adj
        holm[raw[i][0]] = adj
    print("| domain | diff | 95% CI | p | p (Holm) | sig |")
    print("|---" * 6 + "|")
    for d, o, lo_d, hi_d, pv in raw:
        print(f"| {d} | {100 * o:+.2f} | [{100 * lo_d:+.2f}, {100 * hi_d:+.2f}] "
              f"| {pv:.4f} | {holm[d]:.4f} | {'YES' if holm[d] < 0.05 else 'no'} |")
        report["domains"][d] = {"diff": o, "ci95": [lo_d, hi_d], "p": pv, "holm": holm[d]}

    if args.json:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
