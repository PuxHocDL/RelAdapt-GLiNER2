"""Pull finished runs of one launch back into ``results/aste_<domain>/<preset>/``.

``monitor_aste.py`` polls and blocks; this one takes a single pass, writes
whatever has finished and reports what has not, so a sweep spread over several
Modal workspaces can be checked one profile at a time from a shell loop:

    for p in <profile1> <profile2>; do
        GLINER_ABSA_MODAL_PROFILE=$p GLINER_ABSA_DOMAIN=15res \\
        python scripts/collect_runs.py
    done

The payload comes straight from the function call rather than from the results
volume, so a run whose container has exited is still recoverable and no
``modal volume get`` round trip is needed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODAL_PROFILE = os.environ.get("GLINER_ABSA_MODAL_PROFILE", "")
if MODAL_PROFILE:
    os.environ["MODAL_PROFILE"] = MODAL_PROFILE

from modal.functions import FunctionCall  # noqa: E402  (profile must be set first)

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = os.environ.get("GLINER_ABSA_DOMAIN", "15res")
OUT_DIR = ROOT / os.environ.get("GLINER_ABSA_OUT_DIR", f"results/aste_{DOMAIN}")
MANIFEST = os.environ.get("GLINER_ABSA_MANIFEST", "call_manifest.json")


def _from_volume(domain, preset, seed):
    import modal

    try:
        vol = modal.Volume.from_name("gliner2-absa-results")
        data = b"".join(vol.read_file(f"aste/{domain}/{preset}/seed_{seed}/metrics.json"))
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def main() -> int:
    manifest = json.loads((OUT_DIR / MANIFEST).read_text(encoding="utf-8"))
    done, pending, failed = [], [], []

    for key, call in manifest["calls"].items():
        # save_as wins: launch_aste.py records the base preset under "preset" and
        # the directory the run actually writes to under "save_as", so preferring
        # "preset" makes a -fate or -prune run look like the untagged control that
        # is already on disk, and it is then reported done without ever being
        # fetched.
        preset = call.get("save_as") or call.get("preset") or call.get("arm")
        seed = call["seed"]
        dest = OUT_DIR / preset / f"seed_{seed}" / "metrics.json"
        if dest.exists():
            payload = json.loads(dest.read_text(encoding="utf-8"))
            done.append((key, payload["test"]["f1"]))
            continue
        try:
            payload = FunctionCall.from_id(call["call_id"]).get(timeout=0)
        except TimeoutError:
            pending.append(key)
            continue
        except Exception as exc:                      # a crashed run, or an expired result
            # aste_run also writes every payload to the results volume, which does
            # not expire, so a run whose call result is gone is still recoverable.
            payload = _from_volume(call["domain"], preset, seed)
            if payload is None:
                failed.append((key, f"{type(exc).__name__}: {str(exc)[:160]}"))
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")
        done.append((key, payload["test"]["f1"]))

    print(f"[{DOMAIN} @ {MODAL_PROFILE or 'default'}] "
          f"{len(done)} done, {len(pending)} running, {len(failed)} failed")
    for key, f1 in sorted(done):
        print(f"  {key:18} f1={f1:.4f}")
    for key, err in failed:
        print(f"  {key:18} FAILED {err}")
    if pending:
        print(f"  pending: {', '.join(sorted(pending))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
