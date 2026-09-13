"""Per-domain numbers for the paper, computed from saved predictions only.

Single model = mean over seeds of corpus-level P/R/F1 per seed.
Consensus    = keep triplets predicted by >= ceil(tau*K) of K seeds.
"""
import glob
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
DOMAINS = ("14lap", "14res", "15res", "16res")


def load(domain, arm, seeds, split):
    key = "dev_predictions" if split == "dev" else "predictions"
    out = {}
    for p in sorted(glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json"))):
        x = json.loads(Path(p).read_text(encoding="utf-8"))
        s = int(x["seed"])
        if s not in seeds:
            continue
        rows = x.get(key) or []
        out[s] = {r["id"]: ({(tuple(a), tuple(o), q) for a, o, q in r["gold"]},
                            {(tuple(a), tuple(o), q) for a, o, q in r["pred"]}) for r in rows}
    return out


def prf(tp, npd, ng):
    p = tp / npd if npd else 0.0
    r = tp / ng if ng else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def single(per_seed):
    ps, rs, fs = [], [], []
    for s, rows in per_seed.items():
        tp = sum(len(g & y) for g, y in rows.values())
        npd = sum(len(y) for g, y in rows.values())
        ng = sum(len(g) for g, y in rows.values())
        p, r, f = prf(tp, npd, ng)
        ps.append(p); rs.append(r); fs.append(f)
    return st.mean(ps), st.mean(rs), st.mean(fs), (st.stdev(fs) if len(fs) > 1 else 0.0)


def consensus(per_seed, k):
    ids = sorted(next(iter(per_seed.values())))
    tp = npd = ng = 0
    for i in ids:
        gold = next(iter(per_seed.values()))[i][0]
        c = Counter()
        for s in per_seed:
            c.update(per_seed[s][i][1])
        keep = {y for y, n in c.items() if n >= k}
        tp += len(keep & gold); npd += len(keep); ng += len(gold)
    return prf(tp, npd, ng)


def row(label, vals):
    avg = st.mean(vals)
    print(f"| {label} | " + " | ".join(f"{100*v:.2f}" for v in vals) + f" | **{100*avg:.2f}** |")


def table(split):
    print(f"\n## split = {split}\n")
    print("| config | 14lap | 14res | 15res | 16res | avg |")
    print("|---|---:|---:|---:|---:|---:|")
    specs = [
        ("B2 42-46", "B2", range(42, 47)),
        ("B2h 42-46", "B2h", range(42, 47)),
        ("B2h 47-51", "B2h", range(47, 52)),
        ("R1m 42-46", "R1m", range(42, 47)),
        ("R1b 42-46", "R1b", range(42, 47)),
        ("R1bm 42-46", "R1bm", range(42, 47)),
        ("B2h-c2 52-56", "B2h-c2", range(52, 57)),
        ("R1bm-c2 52-56", "R1bm-c2", range(52, 57)),
        ("B2h-cf 62-66", "B2h-cf", range(62, 67)),
        ("R1bm-cf 62-66", "R1bm-cf", range(62, 67)),
        ("R1bm-cs 67-71", "R1bm-cs", range(67, 72)),
        ("R1bm-cs 72-76", "R1bm-cs", range(72, 77)),
        ("R1bm-cs 67-76", "R1bm-cs", range(67, 77)),
    ]
    for label, arm, seeds in specs:
        seeds = set(seeds)
        data = [load(d, arm, seeds, split) for d in DOMAINS]
        if any(len(x) != len(seeds) or not next(iter(x.values()), None) for x in data):
            print(f"| {label} | missing | | | | |")
            continue
        row(label + " single F1", [single(x)[2] for x in data])
        if arm.startswith("R1bm-c"):
            K = len(seeds)
            for tau in (0.6,):
                k = math.ceil(tau * K - 1e-9)
                row(f"{label} consensus c>={k}/{K} F1", [consensus(x, k)[2] for x in data])
                row(f"{label} consensus c>={k}/{K} P", [consensus(x, k)[0] for x in data])
                row(f"{label} consensus c>={k}/{K} R", [consensus(x, k)[1] for x in data])
            row(f"{label} single P", [single(x)[0] for x in data])
            row(f"{label} single R", [single(x)[1] for x in data])
            if K == 10:
                for k in range(1, 11):
                    row(f"{label} c>={k}/10 F1", [consensus(x, k)[2] for x in data])


if __name__ == "__main__":
    for split in sys.argv[1:] or ("dev", "test"):
        table(split)
