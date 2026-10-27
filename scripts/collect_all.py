"""Pull back everything the backbone sweep left on Modal, from any machine, any time.

Every run was spawned detached, so the laptop may be off while they train. This
fetches finished runs (GPU and CPU workspaces), the worst-case speed timing, and
reports what is still running. Re-run until it says nothing is running.

    python scripts/collect_all.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def speed() -> str:
    spec = ROOT / "results/aste_speed/emit_all_call.json"
    dest = ROOT / "results/aste_speed/speed_emit_all.json"
    if dest.exists() or not spec.exists():
        return "speed emit-all: " + ("done" if dest.exists() else "not launched")
    c = json.loads(spec.read_text(encoding="utf-8"))
    child = ("import json,sys\nfrom modal.functions import FunctionCall\n"
             "try:\n    r = FunctionCall.from_id(sys.argv[1]).get(timeout=0)\n"
             "    print(json.dumps(r))\nexcept TimeoutError:\n    print('PENDING')\n")
    r = subprocess.run([sys.executable, "-c", child, c["call_id"]], capture_output=True, text=True,
                       encoding="utf-8", env={**os.environ, "MODAL_PROFILE": c["profile"],
                                              "PYTHONUTF8": "1"})
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    if out.startswith("{"):
        dest.write_text(json.dumps(json.loads(out), indent=2) + "\n", encoding="utf-8")
        return "speed emit-all: done"
    return f"speed emit-all: {out[:120]}"


def main() -> int:
    r = subprocess.run([sys.executable, str(ROOT / "scripts/backbone_sweep.py"), "collect"],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    lines = r.stdout.strip().splitlines()
    print(*[l for l in lines if "FAILED" in l and "cancel" not in l.lower()][:20], sep="\n")
    print(lines[-1] if lines else r.stderr[-500:])
    print(speed())
    return 0


if __name__ == "__main__":
    sys.exit(main())
