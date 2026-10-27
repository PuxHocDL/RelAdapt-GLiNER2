"""Mean and 5-run standard deviation for the single-model rows of Tables I and II.

Per dataset, the std is taken over the per-run corpus F1. For the four-dataset
average, each run's average over the four datasets is computed first (runs are
paired by seed), and the std is taken over those per-run averages. All stds are
sample stds (ddof = 1).

    python analysis/table_std.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("14lap", "14res", "15res", "16res")


def runs(arm: str, seeds: range) -> dict[int, dict[str, tuple[float, float]]]:
    """seed -> domain -> (test F1, dev F1), in percent."""
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for d in DOMAINS:
        for p in glob.glob(str(ROOT / f"results/aste_{d}/{arm}/seed_*/metrics.json")):
            x = json.loads(Path(p).read_text(encoding="utf-8"))
            s = int(x["seed"])
            if s in seeds:
                out.setdefault(s, {})[d] = (100 * x["test"]["f1"], 100 * x["dev_f1"])
    assert len(out) == len(seeds) and all(len(v) == 4 for v in out.values()), (arm, seeds)
    return out


def summary(arm: str, seeds: range) -> dict[str, tuple[float, float]]:
    r = runs(arm, seeds)
    res = {}
    for d in DOMAINS:
        vals = [r[s][d][0] for s in sorted(r)]
        res[d] = (st.mean(vals), st.stdev(vals))
    for key, idx in (("test_avg", 0), ("dev_avg", 1)):
        per_run = [st.mean(r[s][d][idx] for d in DOMAINS) for s in sorted(r)]
        res[key] = (st.mean(per_run), st.stdev(per_run))
    return res


ROWS = [
    # Table I, single-model rows and consensus members
    ("I   generic pair head", "A0", range(42, 47)),
    ("I   GLiNER2-ASTE (canonical)", "B2h-cf", range(62, 67)),
    ("I   RelAdapt-GLiNER2 (canonical) / members A", "R1bm-cf", range(62, 67)),
    ("I   members B", "R1bm-cs", range(67, 72)),
    ("I   members C", "R1bm-cs", range(72, 77)),
    ("I   members B+C", "R1bm-cs", range(67, 77)),
    # Table II
    ("II  screen baseline", "B2h-ex3", range(42, 47)),
    ("II  + deeper MLP + features", "R1m", range(42, 47)),
    ("II  + biaffine", "R1b", range(42, 47)),
    ("II  + biaffine + features", "R1bm", range(42, 47)),
    ("II  replication baseline", "B2h-c2", range(52, 57)),
    ("II  replication RelAdapt-GLiNER2", "R1bm-c2", range(52, 57)),
    ("II  + Sinkhorn prior", "B2", range(42, 47)),
    ("II  + hard-negative weighting", "B2h", range(42, 47)),
]


def main() -> None:
    fmt = lambda t: f"{t[0]:.2f} +- {t[1]:.2f}"  # noqa: E731
    print(f"{'row':46s} | " + " | ".join(f"{d:>14s}" for d in DOMAINS)
          + f" | {'test avg':>14s} | {'dev avg':>14s}")
    for name, arm, seeds in ROWS:
        s = summary(arm, seeds)
        print(f"{name:46s} | " + " | ".join(f"{fmt(s[d]):>14s}" for d in DOMAINS)
              + f" | {fmt(s['test_avg']):>14s} | {fmt(s['dev_avg']):>14s}")


if __name__ == "__main__":
    main()
