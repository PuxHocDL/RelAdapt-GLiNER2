"""Run the inference-cost benchmark on Modal and save it to results/aste_speed/.

    GLINER_ABSA_MODAL_PROFILE=<profile> python scripts/run_speed.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODAL_PROFILE = os.environ.get("GLINER_ABSA_MODAL_PROFILE", "")
if MODAL_PROFILE:
    os.environ["MODAL_PROFILE"] = MODAL_PROFILE

import modal  # noqa: E402  (profile must be set first)

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ["fastino/gliner2-base-v1", "microsoft/deberta-v3-base", "bert-base-uncased"]


def main() -> int:
    emit_all = os.environ.get("GLINER_ABSA_SPEED_EMIT_ALL", "0") == "1"
    fn = modal.Function.from_name("gliner2-absa-research", "aste_speed")
    report = fn.remote(encoder_ids=ENCODERS, batch_sizes=[1, 32],
                       **({"reladapt_only": True, "emit_all": True} if emit_all else {}))
    out = ROOT / "results/aste_speed"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("speed_emit_all.json" if emit_all else "speed.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for r in report["rows"]:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
