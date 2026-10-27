"""Move unfinished base-encoder runs of the backbone sweep from GPU to CPU workspaces.

For every GPU job (except DeBERTa-v3-large, which stays on GPU) each call is
either finished (its result is written to results/), or cancelled and respawned
on a CPU-only deployment (``GLINER_ABSA_GPU=cpu``) of a workspace without a payment
method. The moves are recorded in results/backbone_moves.json, which
``backbone_sweep.py collect`` also reads, so collection can happen at any later
time from any machine.

    python scripts/move_to_cpu.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backbone_sweep as bs  # noqa: E402

# Modal profiles with a CPU-only deployment: GLINER_ABSA_CPU_PROFILES="p1,p2,..."
CPU_PROFILES = tuple(p for p in os.environ.get("GLINER_ABSA_CPU_PROFILES", "").split(",") if p)
GPU_KEEP = ("-enc-dv3l",)
MOVES = ROOT / "results/backbone_moves.json"

CHILD = r'''
import json, os, sys
from pathlib import Path
from modal.functions import FunctionCall
calls = json.loads(sys.argv[1])
out = {}
for key, c in calls.items():
    dest = Path(c["dest"])
    if dest.exists():
        out[key] = "done"; continue
    fc = FunctionCall.from_id(c["call_id"])
    try:
        payload = fc.get(timeout=0)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8")
        out[key] = "done"
    except TimeoutError:
        fc.cancel()
        out[key] = "cancelled"
    except Exception as exc:
        out[key] = "failed"
print(json.dumps(out))
'''


def main() -> int:
    to_move = []      # (domain, preset, model, suffix, seed, lr)
    by_profile = {}
    for profile, preset, model, suffix, seeds, lr, _, domain in bs.jobs():
        if suffix in GPU_KEEP:
            continue
        man = json.loads((ROOT / f"results/aste_{domain}" /
                          bs.manifest_name(preset, suffix, seeds)).read_text(encoding="utf-8"))
        for key, c in man["calls"].items():
            dest = ROOT / f"results/aste_{domain}/{c['save_as']}/seed_{c['seed']}/metrics.json"
            by_profile.setdefault(profile, {})[f"{domain}|{key}"] = {
                "call_id": c["call_id"], "dest": str(dest),
                "job": [domain, preset, model, suffix, c["seed"], lr]}
    for profile, calls in by_profile.items():
        env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONUTF8": "1"}
        r = subprocess.run([sys.executable, "-c", CHILD,
                            json.dumps({k: {"call_id": v["call_id"], "dest": v["dest"]}
                                        for k, v in calls.items()})],
                           env=env, capture_output=True, text=True, encoding="utf-8")
        status = json.loads(r.stdout.strip().splitlines()[-1])
        n = {s: sum(1 for v in status.values() if v == s) for s in ("done", "cancelled", "failed")}
        print(f"[{profile}] {n}", flush=True)
        to_move += [calls[k]["job"] for k, s in status.items() if s != "done"]

    # Respawn round-robin: group by (domain, preset, suffix) so one launch per group
    # and profile keeps manifests small.
    groups = {}
    for i, (domain, preset, model, suffix, seed, lr) in enumerate(sorted(to_move)):
        prof = CPU_PROFILES[i % len(CPU_PROFILES)]
        groups.setdefault((prof, domain, preset, model, suffix, lr), []).append(seed)
    moves = json.loads(MOVES.read_text(encoding="utf-8")) if MOVES.exists() else []
    for (prof, domain, preset, model, suffix, lr), seeds in groups.items():
        seed_s = ",".join(map(str, sorted(seeds)))
        manifest = f"call_manifest{suffix}_{preset}_cpu_{prof.replace(' ', '_')}_{seed_s.replace(',', '-')}.json"
        env = {**bs.env_for(prof, preset, model, suffix, seed_s, lr, domain),
               "GLINER_ABSA_MANIFEST": manifest}
        r = subprocess.run([sys.executable, str(ROOT / "scripts/launch_aste.py")], env=env,
                           capture_output=True, text=True, encoding="utf-8")
        print(f"[{prof}] {preset}{suffix} {domain} seeds {seed_s}:",
              (r.stdout + r.stderr).strip().splitlines()[-1][:80], flush=True)
        moves.append({"profile": prof, "domain": domain, "manifest": manifest})
    MOVES.write_text(json.dumps(moves, indent=1) + "\n", encoding="utf-8")
    print(f"{len(to_move)} runs moved to CPU; recorded in {MOVES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
