"""Backbone sweep: RelAdapt on plain bidirectional encoders, no GLiNER2.

DeBERTa-v3-base becomes the main backbone, so its runs cover everything the paper
reports (consensus groups, framework build-up, relation-matcher screen); the other
encoders get the concatenation matcher and RelAdapt only. All runs use
rng_neutral_load, so every arm on a seed sees the same mini-batch order.

    python scripts/backbone_sweep.py launch
    python scripts/backbone_sweep.py collect
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("14lap", "14res", "15res", "16res")
S_MAIN, S_B, S_C = "62,63,64,65,66", "67,68,69,70,71", "72,73,74,75,76"
DV3, ROB, BERT, DV3L = ("microsoft/deberta-v3-base", "FacebookAI/roberta-base",
                        "bert-base-uncased", "microsoft/deberta-v3-large")
# (preset, encoder, save suffix, seeds, encoder lr or "", cost weight)
ARMS = (
    ("R1bm", DV3, "-enc-dv3", S_B, "", 1),
    ("R1bm", DV3, "-enc-dv3", S_C, "", 1),
    ("A0", DV3, "-enc-dv3", S_MAIN, "", 1),
    ("B2", DV3, "-enc-dv3", S_MAIN, "", 1),
    ("R1b", DV3, "-enc-dv3", S_MAIN, "", 1),
    ("R1m", DV3, "-enc-dv3", S_MAIN, "", 1),
    ("R1bm", ROB, "-enc-rob", S_MAIN, "", 1),
    ("B2h", ROB, "-enc-rob", S_MAIN, "", 1),
    ("B2h", BERT, "-enc-bert", S_MAIN, "", 1),
    ("R1bm", DV3L, "-enc-dv3l", S_MAIN, "1e-5", 3),
    ("B2h", DV3L, "-enc-dv3l", S_MAIN, "1e-5", 3),
)
# Modal profiles that may run GPU functions, with their share of the cost weight:
# GLINER_ABSA_GPU_PROFILES="profileA:24,profileB:24,profileC:12"
PROFILES = {p.rsplit(":", 1)[0]: int(p.rsplit(":", 1)[1])
            for p in os.environ.get("GLINER_ABSA_GPU_PROFILES", "default:1").split(",") if p}


def jobs():
    all_jobs = [(p, m, s, seeds, lr, w, d) for p, m, s, seeds, lr, w in ARMS for d in DOMAINS]
    # Heaviest first, each to the profile with the most budget left.
    all_jobs.sort(key=lambda j: -j[5])
    left = dict(PROFILES)
    out = []
    for j in all_jobs:
        prof = max(left, key=left.get)
        left[prof] -= j[5]
        out.append((prof, *j))
    return out


def manifest_name(preset, suffix, seeds):
    return f"call_manifest{suffix}_{preset}_s{seeds.split(',')[0]}.json"


def env_for(profile, preset, model, suffix, seeds, lr, domain):
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "GLINER_ABSA_MODAL_PROFILE": profile, "GLINER_ABSA_DOMAIN": domain,
            "GLINER_ABSA_PRESETS": preset, "GLINER_ABSA_SEEDS": seeds,
            "GLINER_ABSA_MODEL": model, "GLINER_ABSA_SAVE_SUFFIX": suffix,
            "GLINER_ABSA_RNG_NEUTRAL": "1", "GLINER_ABSA_ENCODER_LR": lr,
            "GLINER_ABSA_MANIFEST": manifest_name(preset, suffix, seeds)}


def main(cmd: str) -> int:
    script = "launch_aste.py" if cmd == "launch" else "collect_runs.py"
    totals = {"done": 0, "running": 0, "failed": 0}
    for profile, preset, model, suffix, seeds, lr, _, domain in jobs():
        manifest = ROOT / f"results/aste_{domain}" / manifest_name(preset, suffix, seeds)
        if cmd == "launch" and manifest.exists():
            print(f"skip {preset}{suffix}/{domain}/{seeds}: already launched", flush=True)
            continue
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           env=env_for(profile, preset, model, suffix, seeds, lr, domain),
                           capture_output=True, text=True, encoding="utf-8")
        lines = (r.stdout + r.stderr).strip().splitlines()
        head = f"[{profile}] {preset}{suffix} {domain} s{seeds[:2]}"
        if cmd == "launch":
            print(head, lines[-1] if lines else "", flush=True)
            continue
        first = lines[0] if lines else ""
        for n, key in re.findall(r"(\d+) (done|running|failed)", first):
            totals[key] += int(n)
        bad = [t for t in lines[1:] if "FAILED" in t]
        print(head, first.split("] ")[-1], *bad[:2], sep="  ", flush=True)
    if cmd != "launch":
        # Runs moved to CPU workspaces by move_to_cpu.py. Their GPU manifests above
        # count them as failed (cancelled); the CPU manifests below hold the real state.
        moves = ROOT / "results/backbone_moves.json"
        import json
        for m in json.loads(moves.read_text(encoding="utf-8")) if moves.exists() else []:
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                   "GLINER_ABSA_MODAL_PROFILE": m["profile"], "GLINER_ABSA_DOMAIN": m["domain"],
                   "GLINER_ABSA_MANIFEST": m["manifest"]}
            r = subprocess.run([sys.executable, str(ROOT / "scripts/collect_runs.py")], env=env,
                               capture_output=True, text=True, encoding="utf-8")
            first = ((r.stdout + r.stderr).strip().splitlines() or [""])[0]
            for n, key in re.findall(r"(\d+) (done|running|failed)", first):
                totals["cpu_" + key] = totals.get("cpu_" + key, 0) + int(n)
            print(f"[{m['profile']}] cpu {m['manifest'][14:60]}", first.split("] ")[-1], flush=True)
        print("TOTAL", totals, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "collect"))
