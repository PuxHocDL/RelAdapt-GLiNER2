"""Poll the ASTE ablation, pull each finished run, then run the significance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Optional: pick a Modal profile; otherwise the active default profile is used.
MODAL_PROFILE = os.environ.get("GLINER_ABSA_MODAL_PROFILE", "")
if MODAL_PROFILE:
    os.environ["MODAL_PROFILE"] = MODAL_PROFILE

import modal
from modal.functions import FunctionCall

ROOT = Path(__file__).resolve().parents[1]
ACSI_DIR = ROOT / os.environ.get("GLINER_ABSA_OUT_DIR", "results/aste_14res")
RESULT_VOLUME = "gliner2-absa-results"
POLL_SECONDS = 60
STALL_LIMIT = 15


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def health(app: str) -> dict:
    try:
        st = modal.Function.from_name(
            app, os.environ.get("GLINER_ABSA_FUNCTION", "aste_run")
        ).get_current_stats()
        return {"backlog": st.backlog, "runners": st.num_total_runners}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def fetch(domain: str, preset: str, seed: int) -> Path:
    dest = ACSI_DIR / preset / f"seed_{seed}" / "metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get("GLINER_ABSA_RESULT_PREFIX", "aste")
    remote = f"{prefix}/{domain}/{preset}/seed_{seed}/metrics.json"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run(
        ["modal", "volume", "get", RESULT_VOLUME, remote, str(dest), "--force"],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return dest


def main() -> int:
    tag = os.environ.get("GLINER_ABSA_TAG", "")
    name = f"call_manifest{'_' + tag if tag else ''}.json"
    manifest = json.loads((ACSI_DIR / name).read_text(encoding="utf-8"))
    calls = manifest["calls"]
    domain = manifest["domain"]
    done: dict[str, str] = {}
    stall = 0
    print(f"[{utc_now()}] monitoring {len(calls)} ASTE runs on {domain}", flush=True)

    while len(done) < len(calls):
        for key, entry in calls.items():
            if key in done:
                continue
            try:
                FunctionCall.from_id(entry["call_id"]).get(timeout=0)
            except TimeoutError:
                continue
            except Exception as exc:
                done[key] = "failed"
                print(f"[{utc_now()}] {key}: FAILED {type(exc).__name__}: {str(exc)[:220]}",
                      flush=True)
                continue
            try:
                path = fetch(domain, entry["preset"], entry["seed"])
                payload = json.loads(path.read_text(encoding="utf-8"))
                t = payload["test"]
                done[key] = "ok"
                if "f1" in t:   # full-ASTE triplet extraction
                    print(f"[{utc_now()}] {key}: tripletF1={t['f1']:.4f} "
                          f"P={t['precision']:.4f} R={t['recall']:.4f}", flush=True)
                else:
                    print(f"[{utc_now()}] {key}: macroF1={t['polarity_macro_f1']:.4f} "
                          f"multi={t['multi_aspect_macro_f1']:.4f} "
                          f"div={t['aspect_repr_divergence']:.4f}", flush=True)
            except Exception as exc:
                print(f"[{utc_now()}] {key}: sync retry ({type(exc).__name__})", flush=True)

        h = health(manifest["app"])
        if h.get("runners") == 0 and len(done) < len(calls):
            stall += 1
            if stall >= STALL_LIMIT:
                print(f"[{utc_now()}] *** no runners for "
                      f"{STALL_LIMIT * POLL_SECONDS // 60} min; stopping ***", flush=True)
                break
        else:
            stall = 0

        print(f"[{utc_now()}] {len(done)}/{len(calls)} | {h}", flush=True)
        if len(done) >= len(calls):
            break
        time.sleep(POLL_SECONDS)

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    stats = subprocess.run(
        [sys.executable, str(ROOT / "research" / "aste_stats.py"), str(ACSI_DIR)],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    print(stats.stdout, flush=True)
    if stats.stderr.strip():
        print(stats.stderr.strip(), file=sys.stderr, flush=True)
    return 0 if len(done) == len(calls) else 1


if __name__ == "__main__":
    raise SystemExit(main())
