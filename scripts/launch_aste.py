"""Spawn the ASTE ablation: presets x seeds on one SemEval domain.

A0 is the mandatory control - span enumeration and pairing with no aspect-
conditioned interaction - so every other configuration is measured against the
plain architecture rather than against nothing.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Optional: pick a Modal profile; otherwise the active default profile is used.
MODAL_PROFILE = os.environ.get("GLINER_ABSA_MODAL_PROFILE", "")
if MODAL_PROFILE:
    os.environ["MODAL_PROFILE"] = MODAL_PROFILE

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "gliner2-absa-research"

DOMAIN = os.environ.get("GLINER_ABSA_DOMAIN", "14res")
_presets = os.environ.get("GLINER_ABSA_PRESETS", "").strip()
PRESETS = tuple(p for p in _presets.split(",") if p) or ("A0", "A1", "A2", "A3", "A4")
_seeds = os.environ.get("GLINER_ABSA_SEEDS", "").strip()
SEEDS = tuple(int(x) for x in _seeds.split(",") if x) or (42, 43, 44, 45, 46)
EPOCHS = int(os.environ.get("GLINER_ABSA_EPOCHS", "20"))
BATCH_SIZE = 8
MODEL_ID = os.environ.get("GLINER_ABSA_MODEL", "fastino/gliner2-base-v1")
# save_as keeps two runs of one preset apart when only the encoder or an
# optimiser setting differs, so results land in separate directories.
SAVE_SUFFIX = os.environ.get("GLINER_ABSA_SAVE_SUFFIX", "").strip()
WARMUP_PCT = float(os.environ.get("GLINER_ABSA_WARMUP_PCT", "0.2"))
LLRD = float(os.environ.get("GLINER_ABSA_LLRD", "1.0"))
GATE_LR = float(os.environ.get("GLINER_ABSA_GATE_LR", "0.0"))
OUT_DIR = ROOT / os.environ.get("GLINER_ABSA_OUT_DIR", f"results/aste_{DOMAIN}")
# collect_runs.py already reads this variable; the launcher ignoring it meant a
# second launch into the same domain silently overwrote the first one's record.
MANIFEST = os.environ.get("GLINER_ABSA_MANIFEST", "call_manifest.json")


def main() -> int:
    fn = modal.Function.from_name(APP_NAME, os.environ.get("GLINER_ABSA_FUNCTION", "aste_run"))
    manifest = {
        "app": APP_NAME,
        "modal_profile": MODAL_PROFILE or "default",
        "domain": DOMAIN,
        "presets": list(PRESETS),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_id": MODEL_ID,
        "save_suffix": SAVE_SUFFIX,
        "warmup_pct": WARMUP_PCT,
        "llrd": LLRD,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "calls": {},
    }
    for preset in PRESETS:
        for seed in SEEDS:
            save_as = f"{preset}{SAVE_SUFFIX}" if SAVE_SUFFIX else ""
            call = fn.spawn(
                domain=DOMAIN, preset=preset, seed=seed,
                model_id=MODEL_ID, epochs=EPOCHS, batch_size=BATCH_SIZE,
                warmup_pct=WARMUP_PCT, llrd=LLRD, gate_lr=GATE_LR, save_as=save_as,
            )
            key = f"{save_as or preset}|seed{seed}"
            manifest["calls"][key] = {
                "call_id": call.object_id, "preset": preset, "save_as": save_as,
                "seed": seed, "domain": DOMAIN,
            }
            print(f"spawned {key} -> {call.object_id}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{len(manifest['calls'])} calls -> {OUT_DIR / MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
