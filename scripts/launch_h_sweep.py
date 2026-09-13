"""Spread the H-series sweep across several Modal workspaces.

``launch_aste.py`` spawns one (preset x seed) grid onto one profile. The H ladder
is 4 domains x 6 arms x 5 seeds and no single free-tier workspace has the GPU
quota for it, so this shards the grid by profile and writes one manifest per
profile per domain, in the same format ``collect_runs.py`` already reads.

Sharding is by **seed**, never by arm: every workspace then runs the same set of
configurations, so if one workspace turns out to be slower, dies mid-sweep or
gets a different GPU, it costs seeds uniformly across arms instead of wiping out
a whole arm and making the ladder unreadable.

    GLINER_ABSA_PROFILES=<profile1>,<profile2>,<profile3> \
    GLINER_ABSA_DOMAINS=15res,14lap \
    GLINER_ABSA_PRESETS=H1,H2,H3 \
    python scripts/launch_h_sweep.py

Nothing here is H-specific except the defaults; it will shard any preset list.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "gliner2-absa-research"

# Comma-separated Modal profile names; required, there is no sensible default.
PROFILES = tuple(p for p in os.environ.get("GLINER_ABSA_PROFILES", "").split(",") if p)
DOMAINS = tuple(d for d in os.environ.get(
    "GLINER_ABSA_DOMAINS", "15res,14lap").split(",") if d)
PRESETS = tuple(p for p in os.environ.get(
    "GLINER_ABSA_PRESETS", "H1,H2,H3").split(",") if p)
SEEDS = tuple(int(s) for s in os.environ.get(
    "GLINER_ABSA_SEEDS", "42,43,44,45,46").split(",") if s)
EPOCHS = int(os.environ.get("GLINER_ABSA_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("GLINER_ABSA_BATCH_SIZE", "8"))
MODEL_ID = os.environ.get("GLINER_ABSA_MODEL", "fastino/gliner2-base-v1")
WARMUP_PCT = float(os.environ.get("GLINER_ABSA_WARMUP_PCT", "0.2"))
SAVE_SUFFIX = os.environ.get("GLINER_ABSA_SAVE_SUFFIX", "").strip()
MANIFEST = os.environ.get("GLINER_ABSA_MANIFEST", "call_manifest_h.json")
DRY_RUN = os.environ.get("GLINER_ABSA_DRY_RUN", "").strip() not in ("", "0", "false")
# Set by the parent process when it re-enters this script once per profile.
SHARD = os.environ.get("GLINER_ABSA_SHARD_PROFILE", "").strip()


def plan() -> dict:
    """(profile, domain) -> [(preset, seed), ...], sharded round-robin by seed."""
    out = defaultdict(list)
    for domain in DOMAINS:
        for i, seed in enumerate(SEEDS):
            profile = PROFILES[i % len(PROFILES)]
            for preset in PRESETS:
                out[(profile, domain)].append((preset, seed))
    return out


def main() -> int:
    if not PROFILES:
        print("set GLINER_ABSA_PROFILES to a comma-separated list of Modal profiles")
        return 2
    grid = plan()
    total = sum(len(v) for v in grid.values())
    if SHARD:
        total = sum(len(v) for k, v in grid.items() if k[0] == SHARD)
    print(f"{len(PRESETS)} presets x {len(DOMAINS)} domains x {len(SEEDS)} seeds "
          f"= {total} runs over {len(PROFILES)} profiles\n")
    for (profile, domain), jobs in sorted(grid.items()):
        if SHARD and profile != SHARD:
            continue
        seeds = sorted({s for _, s in jobs})
        print(f"  {profile:14} {domain:6} {len(jobs):3d} runs   seeds {seeds}")
    if DRY_RUN:
        print("\nGLINER_ABSA_DRY_RUN set - nothing spawned")
        return 0

    print()
    # One subprocess per profile. Setting MODAL_PROFILE between spawns in a single
    # process does nothing: modal resolves the profile when it is first imported,
    # so every call after the first silently lands on whichever workspace was
    # active at import time. That failure is invisible - the spawns all succeed -
    # and the only symptom is one workspace holding the entire backlog.
    if not SHARD:
        import subprocess
        rc = 0
        for profile in PROFILES:
            env = {**os.environ, "MODAL_PROFILE": profile,
                   "GLINER_ABSA_SHARD_PROFILE": profile, "PYTHONUTF8": "1"}
            print(f"--- spawning {profile} ---", flush=True)
            rc |= subprocess.call([sys.executable, __file__], env=env)
        return rc

    manifests = defaultdict(lambda: {"app": APP_NAME, "calls": {}})
    import modal
    for (profile, domain), jobs in sorted(grid.items()):
        if profile != SHARD:
            continue
        fn = modal.Function.from_name(APP_NAME, "aste_run")
        for preset, seed in jobs:
            save_as = f"{preset}{SAVE_SUFFIX}" if SAVE_SUFFIX else ""
            call = fn.spawn(
                domain=domain, preset=preset, seed=seed, model_id=MODEL_ID,
                epochs=EPOCHS, batch_size=BATCH_SIZE, warmup_pct=WARMUP_PCT,
                save_as=save_as,
            )
            key = f"{save_as or preset}|seed{seed}"
            m = manifests[(profile, domain)]
            m.update({"modal_profile": profile, "domain": domain,
                      "presets": list(PRESETS), "seeds": seeds_of(jobs),
                      "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                      "model_id": MODEL_ID, "warmup_pct": WARMUP_PCT,
                      "launched_at": datetime.now(timezone.utc).isoformat()})
            m["calls"][key] = {"call_id": call.object_id, "preset": preset,
                               "save_as": save_as, "seed": seed, "domain": domain,
                               "profile": profile}
            print(f"  {profile:14} {domain:6} {key:16} -> {call.object_id}", flush=True)

    written = []
    for (profile, domain), m in manifests.items():
        out_dir = ROOT / f"results/aste_{domain}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{Path(MANIFEST).stem}_{profile}.json"
        path.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        written.append(path)
    print(f"\n{total} calls spawned; {len(written)} manifests written")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")
    return 0


def seeds_of(jobs) -> list[int]:
    return sorted({s for _, s in jobs})


if __name__ == "__main__":
    sys.exit(main())
