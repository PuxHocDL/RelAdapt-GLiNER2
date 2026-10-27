"""One-epoch wiring check of the encoder-control arm before the full sweep.

    GLINER_ABSA_MODAL_PROFILE=<profile> python scripts/smoke_encoder.py
"""

from __future__ import annotations

import os
import sys

MODAL_PROFILE = os.environ.get("GLINER_ABSA_MODAL_PROFILE", "")
if MODAL_PROFILE:
    os.environ["MODAL_PROFILE"] = MODAL_PROFILE

import modal  # noqa: E402  (profile must be set first)

MODELS = tuple(m for m in os.environ.get(
    "GLINER_ABSA_SMOKE_MODELS", "microsoft/deberta-v3-base,bert-base-uncased").split(",") if m)


def main() -> int:
    fn = modal.Function.from_name("gliner2-absa-research", "aste_run")
    calls = {m: fn.spawn(domain="15res", preset="R1bm", seed=62, model_id=m, epochs=1,
                         warmup_pct=0.2, save_as=f"smoke-{m.split('/')[-1]}")
             for m in MODELS}
    for m, c in calls.items():
        try:
            r = c.get(timeout=1800)
        except Exception as exc:
            print(f"{m}: FAILED {type(exc).__name__}: {str(exc)[:600]}")
            continue
        print(f"{m}: type={r.get('encoder_type')} params={r['params']} "
              f"data={r['data_stats']} dev_f1={r['dev_f1']:.4f} "
              f"test_f1={r['test']['f1']:.4f} hash={r['batch_order_hash']} gpu={r['gpu']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
