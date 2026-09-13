"""Bootstrap CIs, ensemble-size curve and decoder funnel for the paper. Read-only."""
import glob
import itertools
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # project root
OUT = Path(__file__).with_name("paper_stats.json")  # read by make_figures.py
DOMAINS = ("14lap", "14res", "15res", "16res")
N_BOOT = 10000
rng = np.random.default_rng(20260913)


def payloads(domain, arm, seeds):
    out = {}
    for p in glob.glob(str(ROOT / f"results/aste_{domain}/{arm}/seed_*/metrics.json")):
        x = json.loads(Path(p).read_text(encoding="utf-8"))
        if int(x["seed"]) in seeds:
            out[int(x["seed"])] = x
    assert len(out) == len(seeds), (domain, arm, seeds, len(out))
    return out


def sets(x, split):
    key = "dev_predictions" if split == "dev" else "predictions"
    rows = sorted(x[key], key=lambda r: r["id"])
    return [({(tuple(a), tuple(o), q) for a, o, q in r["gold"]},
             {(tuple(a), tuple(o), q) for a, o, q in r["pred"]}) for r in rows]


def f1_from(c):  # c (..., 3) summed counts
    tp, npd, ng = c[..., 0], c[..., 1], c[..., 2]
    p = np.divide(tp, npd, out=np.zeros_like(tp, dtype=float), where=npd > 0)
    r = np.divide(tp, ng, out=np.zeros_like(tp, dtype=float), where=ng > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def seed_counts(rows):
    return np.array([[len(g & y), len(y), len(g)] for g, y in rows], dtype=float)


def cons_counts(rows_by_seed, k):
    n = len(next(iter(rows_by_seed.values())))
    out = np.zeros((n, 3))
    for i in range(n):
        gold = next(iter(rows_by_seed.values()))[i][0]
        cnt = {}
        for rows in rows_by_seed.values():
            for y in rows[i][1]:
                cnt[y] = cnt.get(y, 0) + 1
        keep = {y for y, c in cnt.items() if c >= k}
        out[i] = (len(keep & gold), len(keep), len(gold))
    return out


def boot_arm_diff(ctrl, treat, seeds, split):
    """mean over domains of (mean-over-seeds F1 treat - ctrl), sentence bootstrap."""
    obs, dist, per = [], np.zeros(N_BOOT), {}
    for d in DOMAINS:
        C, T = payloads(d, ctrl, seeds), payloads(d, treat, seeds)
        cc = np.stack([seed_counts(sets(C[s], split)) for s in sorted(seeds)])  # S,n,3
        tc = np.stack([seed_counts(sets(T[s], split)) for s in sorted(seeds)])
        n = cc.shape[1]
        o = f1_from(tc.sum(1)).mean() - f1_from(cc.sum(1)).mean()
        idx = rng.integers(0, n, size=(N_BOOT, n))
        bt = np.stack([f1_from(tc[s][idx].sum(1)) for s in range(tc.shape[0])]).mean(0)
        bc = np.stack([f1_from(cc[s][idx].sum(1)) for s in range(cc.shape[0])]).mean(0)
        dist += (bt - bc) / len(DOMAINS)
        obs.append(o)
        per[d] = 100 * o
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return dict(diff=100 * np.mean(obs), ci=[100 * lo, 100 * hi],
                p_one_sided=float((dist <= 0).mean()), per_domain=per)


def boot_consensus(arm, seeds, split, k):
    obs, dist, per = [], np.zeros(N_BOOT), {}
    for d in DOMAINS:
        P = payloads(d, arm, seeds)
        rows = {s: sets(P[s], split) for s in sorted(seeds)}
        mc = np.stack([seed_counts(rows[s]) for s in sorted(seeds)])
        cc = cons_counts(rows, k)
        n = cc.shape[0]
        o = f1_from(cc.sum(0)) - f1_from(mc.sum(1)).mean()
        idx = rng.integers(0, n, size=(N_BOOT, n))
        b = f1_from(cc[idx].sum(1)) - np.stack([f1_from(mc[s][idx].sum(1)) for s in range(len(seeds))]).mean(0)
        dist += b / len(DOMAINS)
        obs.append(float(o))
        per[d] = 100 * float(o)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return dict(diff=100 * np.mean(obs), ci=[100 * lo, 100 * hi],
                p_one_sided=float((dist <= 0).mean()), per_domain=per)


def size_curve(arm, seeds, split, tau=0.6):
    seeds = sorted(seeds)
    dom = {}
    for d in DOMAINS:
        P = payloads(d, arm, set(seeds))
        rows = {s: sets(P[s], split) for s in seeds}
        n = len(rows[seeds[0]])
        items = []  # per sentence: n_gold, list of (mask, is_tp)
        for i in range(n):
            gold = rows[seeds[0]][i][0]
            m = {}
            for b, s in enumerate(seeds):
                for y in rows[s][i][1]:
                    m[y] = m.get(y, 0) | (1 << b)
            items.append((len(gold), [(mask, y in gold) for y, mask in m.items()]))
        dom[d] = items
    ng = {d: sum(g for g, _ in dom[d]) for d in DOMAINS}
    curve = {}
    for K in range(1, len(seeds) + 1):
        k = max(1, math.ceil(tau * K - 1e-9))
        fs, ps, rs = [], [], []
        for combo in itertools.combinations(range(len(seeds)), K):
            cm = sum(1 << b for b in combo)
            df, dp, dr = [], [], []
            for d in DOMAINS:
                tp = npd = 0
                for _, cand in dom[d]:
                    for mask, is_tp in cand:
                        if bin(mask & cm).count("1") >= k:
                            npd += 1
                            tp += is_tp
                p = tp / npd if npd else 0
                r = tp / ng[d]
                df.append(2 * p * r / (p + r) if p + r else 0); dp.append(p); dr.append(r)
            fs.append(st.mean(df)); ps.append(st.mean(dp)); rs.append(st.mean(dr))
        curve[K] = dict(k=k, n_combos=len(fs), f1=100 * st.mean(fs),
                        f1_sd=100 * (st.pstdev(fs) if len(fs) > 1 else 0.0),
                        p=100 * st.mean(ps), r=100 * st.mean(rs))
        print(split, K, curve[K], flush=True)
    return curve


def funnel(arm, seeds, split="dev"):
    tot = {}
    for d in DOMAINS:
        for x in payloads(d, arm, seeds).values():
            for k, v in x[split]["funnel"].items():
                tot[k] = tot.get(k, 0) + v
    n = tot.pop("n_gold")
    return {k: 100 * v / n for k, v in tot.items()} | {"n_gold": n}


if __name__ == "__main__":
    S62, S67, S72, S10 = set(range(62, 67)), set(range(67, 72)), set(range(72, 77)), set(range(67, 77))
    res = {}
    for split in ("dev", "test"):
        res[f"r1bm_vs_b2h_cf_{split}"] = boot_arm_diff("B2h-cf", "R1bm-cf", S62, split)
        print(split, "R1bm-cf vs B2h-cf", res[f"r1bm_vs_b2h_cf_{split}"], flush=True)
        for name, arm, s, k in (("cons5_62", "R1bm-cf", S62, 3), ("cons5_67", "R1bm-cs", S67, 3),
                                ("cons5_72", "R1bm-cs", S72, 3), ("cons10", "R1bm-cs", S10, 6)):
            res[f"{name}_{split}"] = boot_consensus(arm, s, split, k)
            print(split, name, res[f"{name}_{split}"], flush=True)
    for arm in ("B2h-cf", "R1bm-cf"):
        res[f"funnel_{arm}"] = funnel(arm, S62)
        print(arm, res[f"funnel_{arm}"])
    for split in ("dev", "test"):
        res[f"size_curve_{split}"] = size_curve("R1bm-cs", S10, split)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)
