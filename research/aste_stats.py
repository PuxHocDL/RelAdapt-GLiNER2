"""Significance testing for the ASTE triplet-F1 ablation.

Triplet F1 is a corpus-level, set-based metric, so it cannot be averaged per
item. The paired bootstrap therefore resamples *sentences* and recomputes
corpus P/R/F1 from the tp / n_pred / n_gold counts of the resampled sentences,
separately for every seed, then averages. That captures sentence sampling noise
and seed noise together.

A seed-level t-test is also reported, but with five seeds it is underpowered by
construction and is not what the conclusions rest on.

Usage:
    python research/aste_stats.py results/aste_14res [control_preset]
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

N_BOOTSTRAP = 10000
DEFAULT_CONTROL = "A0"


def load_runs(root: str | Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(Path(root).glob("*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(payload["preset"], {})[int(payload["seed"])] = payload
    return runs


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def _counts(payload: dict) -> dict[str, tuple[int, int, int]]:
    return {r["id"]: (r["tp"], r["n_pred"], r["n_gold"]) for r in payload["predictions"]}


def _corpus_f1(mat: np.ndarray, idx: np.ndarray) -> float:
    tp, n_pred, n_gold = mat[0][idx].sum(), mat[1][idx].sum(), mat[2][idx].sum()
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def bootstrap_test(runs, a: str, b: str, n_boot: int = N_BOOTSTRAP, rng_seed: int = 0) -> dict:
    seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    if not seeds:
        return {"error": "no shared seeds"}
    ca = [_counts(runs[a][s]) for s in seeds]
    cb = [_counts(runs[b][s]) for s in seeds]
    keys = sorted(set(ca[0]).intersection(*[set(d) for d in ca[1:] + cb]))
    if len(keys) < 10:
        return {"error": f"only {len(keys)} shared sentences"}

    A = np.array([[[d[k][i] for k in keys] for i in range(3)] for d in ca], dtype=float)
    B = np.array([[[d[k][i] for k in keys] for i in range(3)] for d in cb], dtype=float)
    n = len(keys)
    idx = np.arange(n)
    score = lambda M, ix: float(np.mean([_corpus_f1(m, ix) for m in M]))  # noqa: E731

    observed = score(B, idx) - score(A, idx)
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, n, n)
        diffs[i] = score(B, sample) - score(A, sample)

    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "n_sentences": n, "n_seeds": len(seeds), "observed_diff": observed,
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_value": float(min(1.0, p)), "n_bootstrap": n_boot,
    }


def seed_test(runs, a: str, b: str) -> dict:
    seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    xa = [runs[a][s]["test"]["f1"] for s in seeds]
    xb = [runs[b][s]["test"]["f1"] for s in seeds]
    diffs = [y - x for x, y in zip(xa, xb)]
    if len(diffs) < 2 or all(d == diffs[0] for d in diffs):
        return {"n_seeds": len(diffs), "p_value": None}
    sd = statistics.stdev(diffs)
    t = statistics.fmean(diffs) / (sd / math.sqrt(len(diffs)))
    df = len(diffs) - 1
    return {"n_seeds": len(diffs), "t": t, "cohens_d": statistics.fmean(diffs) / sd,
            "p_value": float(_betainc(df / 2, 0.5, df / (df + t * t)))}


def holm(pvalues: dict[str, float]) -> dict[str, dict]:
    valid = {k: v for k, v in pvalues.items() if v is not None}
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    m, out, running = len(ordered), {}, 0.0
    for i, (key, p) in enumerate(ordered):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[key] = {"p_raw": p, "p_holm": adj, "significant_0.05": adj < 0.05}
    for key in pvalues:
        out.setdefault(key, {"p_raw": None, "p_holm": None, "significant_0.05": False})
    return out


def analyse(root: str | Path, control: str = DEFAULT_CONTROL) -> dict:
    runs = load_runs(root)
    if control not in runs:
        raise ValueError(f"control {control} missing; have {sorted(runs)}")

    summary = {}
    for preset, by_seed in sorted(runs.items()):
        f1 = [p["test"]["f1"] for p in by_seed.values()]
        summary[preset] = {
            "n_seeds": len(f1),
            "f1_mean": statistics.fmean(f1),
            "f1_std": statistics.stdev(f1) if len(f1) > 1 else 0.0,
            "precision": statistics.fmean(p["test"]["precision"] for p in by_seed.values()),
            "recall": statistics.fmean(p["test"]["recall"] for p in by_seed.values()),
            "per_seed": {s: p["test"]["f1"] for s, p in sorted(by_seed.items())},
            "added_params": next(iter(by_seed.values()))["params"]["added"],
            "pct_of_encoder": next(iter(by_seed.values()))["params"]["added_pct_of_encoder"],
        }

    comparisons = {}
    for preset in sorted(runs):
        if preset == control:
            continue
        comparisons[f"{control}->{preset}"] = {
            "bootstrap": bootstrap_test(runs, control, preset),
            "seed_test": seed_test(runs, control, preset),
        }
    for key, adj in holm({k: v["bootstrap"].get("p_value")
                          for k, v in comparisons.items()}).items():
        comparisons[key]["holm"] = adj

    return {"root": str(root), "control": control,
            "summary": summary, "comparisons": comparisons}


def render(report: dict) -> str:
    lines = [
        f"# ASTE ablation - {report['root']}",
        "", f"Control: `{report['control']}`  |  metric: exact triplet F1", "",
        "| Config | n | F1 | Std | P | R | Added params | % enc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for preset, s in report["summary"].items():
        lines.append(
            f"| `{preset}` | {s['n_seeds']} | **{s['f1_mean']:.4f}** | {s['f1_std']:.4f} "
            f"| {s['precision']:.4f} | {s['recall']:.4f} | {s['added_params']:,} "
            f"| {s['pct_of_encoder']}% |"
        )
    lines += ["", "## Against the control", "",
              "| Comparison | Diff | Bootstrap 95% CI | p | p (Holm) | Sig |",
              "|---|---:|---|---:|---:|:--:|"]
    for key, c in report["comparisons"].items():
        b, h = c["bootstrap"], c["holm"]
        if "error" in b:
            lines.append(f"| {key} | - | {b['error']} | - | - | - |")
            continue
        lines.append(
            f"| {key} | {b['observed_diff']:+.4f} "
            f"| [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['p_value']:.4f} "
            f"| {h['p_holm']:.4f} | {'YES' if h['significant_0.05'] else 'no'} |"
        )
    lines += ["", f"Paired bootstrap: {N_BOOTSTRAP:,} resamples of test sentences, corpus",
              "F1 recomputed per seed and averaged.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results/aste_14res"
    control = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONTROL
    rep = analyse(root, control)
    Path(root, "significance.json").write_text(
        json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8")
    md = render(rep)
    Path(root, "significance.md").write_text(md, encoding="utf-8")
    print(md)
