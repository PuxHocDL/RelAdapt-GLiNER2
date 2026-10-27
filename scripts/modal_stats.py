"""Backlog and runner count of aste_run on each workspace (stall detector).

    python scripts/modal_stats.py <profile> [<profile> ...]
"""

from __future__ import annotations

import os
import subprocess
import sys

CHILD = (
    "import modal\n"
    "fn = modal.Function.from_name('gliner2-absa-research', 'aste_run')\n"
    "s = fn.get_current_stats()\n"
    "print(f'backlog={s.backlog} runners={s.num_total_runners}')\n"
)

for profile in sys.argv[1:]:
    env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True,
                       text=True, encoding="utf-8", timeout=120)
    print(f"{profile:22s}", (r.stdout.strip() or r.stderr.strip()[-200:]))
