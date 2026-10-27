"""Encoder control: is RelAdapt's result the formulation, or GLiNER2's pretraining?

Every arm uses seeds 62-66 and rng_neutral_load, so all backbones on one seed see
the same mini-batch order (check ``batch_order_hash``).

    python scripts/encoder_control.py launch     # spawn every job once
    python scripts/encoder_control.py collect    # pull finished runs, report status
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = "62,63,64,65,66"
DOMAINS = ("14lap", "14res", "15res", "16res")
# (preset, encoder, save suffix)
ARMS = (
    ("R1bm", "fastino/gliner2-base-v1", "-enc-gl2"),
    ("R1bm", "microsoft/deberta-v3-base", "-enc-dv3"),
    ("R1bm", "bert-base-uncased", "-enc-bert"),
    ("B2h", "microsoft/deberta-v3-base", "-enc-dv3"),
)
# Modal profiles that may run GPU functions: GLINER_ABSA_GPU_PROFILES="p1,p2,p3".
PROFILES = tuple(p for p in os.environ.get("GLINER_ABSA_GPU_PROFILES", "default").split(",") if p)
# Jobs per profile, 16 jobs of 5 seeds in total, spread as evenly as possible.
SHARE = tuple(16 // len(PROFILES) + (i < 16 % len(PROFILES)) for i in range(len(PROFILES)))


def jobs():
    out = [(p, m, s, d) for p, m, s in ARMS for d in DOMAINS]
    assigned, i = [], 0
    for prof, n in zip(PROFILES, SHARE):
        for job in out[i:i + n]:
            assigned.append((prof, *job))
        i += n
    assert i == len(out)
    return assigned


def env_for(profile, preset, model, suffix, domain):
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "GLINER_ABSA_MODAL_PROFILE": profile, "GLINER_ABSA_DOMAIN": domain,
            "GLINER_ABSA_PRESETS": preset, "GLINER_ABSA_SEEDS": SEEDS,
            "GLINER_ABSA_MODEL": model, "GLINER_ABSA_SAVE_SUFFIX": suffix,
            "GLINER_ABSA_RNG_NEUTRAL": "1",
            "GLINER_ABSA_MANIFEST": f"call_manifest{suffix}_{preset}.json"}


def main(cmd: str) -> int:
    script = "launch_aste.py" if cmd == "launch" else "collect_runs.py"
    for profile, preset, model, suffix, domain in jobs():
        manifest = ROOT / f"results/aste_{domain}/call_manifest{suffix}_{preset}.json"
        if cmd == "launch" and manifest.exists():
            print(f"skip {preset}{suffix}/{domain}: already launched", flush=True)
            continue
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           env=env_for(profile, preset, model, suffix, domain),
                           capture_output=True, text=True, encoding="utf-8")
        tail = (r.stdout + r.stderr).strip().splitlines()
        head = f"[{profile}] {preset}{suffix} {domain}"
        if cmd == "launch":
            print(head, tail[-1] if tail else "", flush=True)
        else:
            print(head, *tail[:1], *[t for t in tail[1:] if "FAILED" in t or "pending" in t],
                  sep="\n  ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "collect"))
