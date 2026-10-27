"""Modal app for the ASTE triplet-extraction experiments.

Holds the full-ASTE runs (``aste_run``/``aste_smoke``) plus a few GLiNER2
introspection helpers used when wiring a new head onto the pretrained encoder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

GPU_TYPE = os.environ.get("GLINER_ABSA_GPU", "A100")
# GLINER_ABSA_GPU=cpu deploys a CPU-only app: since 2026-10 Modal refuses every GPU
# type on workspaces without a payment method, but CPU still runs on their credit.
if GPU_TYPE.lower() in ("cpu", "none", ""):
    GPU_TYPE = None
RUN_CPU = float(os.environ.get("GLINER_ABSA_CPU", "8" if GPU_TYPE else "16"))
RUN_TIMEOUT = 14400 if GPU_TYPE else 86400

app = modal.App("gliner2-absa-research")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gliner2[local]==2.0.0",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
        "scikit-learn==1.7.2",
    )
    # Separate layer so the cached gliner2 layer is reused; DeBERTa-v3's fast
    # tokenizer is converted from its sentencepiece model on first load.
    .pip_install("sentencepiece==0.2.0", "protobuf==5.29.4")
    .add_local_dir("Data", remote_path="/root/Data")
    .add_local_dir("research", remote_path="/root/research")
)

results_vol = modal.Volume.from_name("gliner2-absa-results", create_if_missing=True)
hf_vol = modal.Volume.from_name("gliner2-absa-hf", create_if_missing=True)

BASE_MODEL = "fastino/gliner2-base-v1"


@app.function(image=image, gpu=GPU_TYPE, timeout=1800, volumes={"/root/.cache/huggingface": hf_vol})
def introspect(model_id: str = BASE_MODEL) -> dict:
    """Map out GLiNER2 internals so ACSI can hook the encoder correctly."""
    import inspect

    import torch
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(model_id)
    report: dict = {"model_id": model_id, "model_class": type(model).__name__}

    report["top_level_attrs"] = sorted(
        a for a in dir(model) if not a.startswith("_") and not callable(getattr(model, a, None))
    )
    report["methods"] = sorted(
        a for a in dir(model) if not a.startswith("_") and callable(getattr(model, a, None))
    )

    # Named submodules, one level deep, with parameter counts.
    children = {}
    for name, module in model.named_children():
        children[name] = {
            "class": type(module).__name__,
            "params": sum(p.numel() for p in module.parameters()),
            "children": [f"{n}:{type(m).__name__}" for n, m in module.named_children()][:20],
        }
    report["named_children"] = children
    report["total_params"] = sum(p.numel() for p in model.parameters())

    # Anything that looks like the transformer encoder.
    for attr in ("encoder", "transformer", "backbone", "model", "bert", "deberta"):
        obj = getattr(model, attr, None)
        if obj is not None and isinstance(obj, torch.nn.Module):
            report[f"attr_{attr}"] = {
                "class": type(obj).__name__,
                "params": sum(p.numel() for p in obj.parameters()),
                "children": [f"{n}:{type(m).__name__}" for n, m in obj.named_children()][:20],
            }

    processor = getattr(model, "processor", None)
    if processor is not None:
        report["processor_class"] = type(processor).__name__
        report["processor_methods"] = sorted(
            a for a in dir(processor) if not a.startswith("_") and callable(getattr(processor, a, None))
        )
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            report["tokenizer_class"] = type(tok).__name__
            report["special_tokens"] = {
                "all_special_tokens": list(getattr(tok, "all_special_tokens", []))[:40],
                "vocab_size": getattr(tok, "vocab_size", None),
            }
            for probe in ("[P]", "[E]", "[C]", "[L]", "[SEP]"):
                try:
                    report.setdefault("schema_token_ids", {})[probe] = tok.convert_tokens_to_ids(probe)
                except Exception as exc:
                    report.setdefault("schema_token_ids", {})[probe] = f"err: {exc}"

    for fname in ("forward", "extract_entities", "classify_text", "create_schema"):
        fn = getattr(model, fname, None)
        if fn is not None:
            try:
                report.setdefault("signatures", {})[fname] = str(inspect.signature(fn))
            except Exception as exc:
                report.setdefault("signatures", {})[fname] = f"err: {exc}"

    if processor is not None:
        for fname in ("collate_fn_train", "prepare_inputs", "prepare_batch"):
            fn = getattr(processor, fname, None)
            if fn is not None:
                try:
                    report.setdefault("signatures", {})[f"processor.{fname}"] = str(
                        inspect.signature(fn)
                    )
                except Exception as exc:
                    report.setdefault("signatures", {})[f"processor.{fname}"] = f"err: {exc}"

    print(json.dumps(report, indent=2, default=str)[:12000], flush=True)
    return report


@app.function(image=image, gpu=GPU_TYPE, timeout=1800, volumes={"/root/.cache/huggingface": hf_vol})
def source_dump() -> dict:
    """Return the gliner2 source files that matter for hooking the encoder."""
    import gliner2

    root = Path(gliner2.__file__).parent
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py"))
    out = {"gliner2_root": str(root), "files": files}
    print(json.dumps(out, indent=2)[:6000], flush=True)
    return out


@app.function(image=image, gpu=GPU_TYPE, timeout=1800, volumes={"/root/.cache/huggingface": hf_vol})
def read_source(relative: str, max_chars: int = 60000) -> str:
    """Read one gliner2 source file so the ACSI wrapper matches the real API."""
    import gliner2

    path = Path(gliner2.__file__).parent / relative
    text = path.read_text(encoding="utf-8")[:max_chars]
    print(text, flush=True)
    return text


@app.function(
    image=image, gpu=GPU_TYPE, cpu=RUN_CPU, memory=32768, timeout=RUN_TIMEOUT,
    max_containers=10,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_vol},
)
def aste_run(domain: str = "14res", preset: str = "A0", seed: int = 42,
             model_id: str = BASE_MODEL, epochs: int = 20,
             batch_size: int = 8, save_as: str = "",
             encoder_lr: float = 2e-5, head_lr: float = 1e-4,
             warmup_pct: float = 0.1, llrd: float = 1.0,
             gate_lr: float = 0.0, rng_neutral_load: bool = False) -> dict:
    """One full-ASTE configuration x seed: extract triplets, score exact F1."""
    import sys

    sys.path.insert(0, "/root/research")
    from aste_full_train import run_one

    payload = run_one(domain=domain, preset=preset, seed=seed, model_id=model_id,
                      epochs=epochs, batch_size=batch_size, data_root="/root/Data/ASTE",
                      encoder_lr=encoder_lr, head_lr=head_lr, warmup_pct=warmup_pct,
                      llrd=llrd, gate_lr=gate_lr, rng_neutral_load=rng_neutral_load)
    if save_as:
        payload["preset"] = save_as
        payload["base_preset"] = preset
    # save_as keeps runs that share a preset but differ in encoder from colliding.
    out = Path(f"/results/aste/{domain}/{save_as or preset}/seed_{seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + chr(10),
        encoding="utf-8",
    )
    results_vol.commit()
    t = payload["test"]
    print(f"[{preset}/{domain}/seed{seed}] triplet P={t['precision']:.4f} "
          f"R={t['recall']:.4f} F1={t['f1']:.4f} (dev {payload['dev_f1']:.4f}) "
          f"added={payload['params']['added']:,}", flush=True)
    return payload


@app.function(image=image, gpu=GPU_TYPE, cpu=8, timeout=7200,
              volumes={"/root/.cache/huggingface": hf_vol})
def aste_smoke(domain: str = "14res", epochs: int = 3) -> dict:
    """Short run of every ASTE preset to catch wiring errors before the sweep."""
    import sys

    sys.path.insert(0, "/root/research")
    from aste_full_train import run_one

    report = {}
    for preset in ("A0", "A1", "A2", "A3", "A4"):
        try:
            r = run_one(domain=domain, preset=preset, seed=42, epochs=epochs,
                        data_root="/root/Data/ASTE")
            t = r["test"]
            report[preset] = {"ok": True, "f1": t["f1"], "p": t["precision"],
                              "r": t["recall"], "n_pred": t["n_pred"],
                              "n_gold": t["n_gold"],
                              "added": r["params"]["added"],
                              "pct": r["params"]["added_pct_of_encoder"]}
        except Exception as exc:
            report[preset] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        print(preset, report[preset], flush=True)
    return report




@app.function(
    image=image, gpu=GPU_TYPE, cpu=8, memory=32768, timeout=21600,
    max_containers=10,
    volumes={"/results": results_vol, "/root/.cache/huggingface": hf_vol},
)
def aste_fusion_run(domain: str = "15res", arm: str = "F1", seed: int = 42,
                    model_id: str = BASE_MODEL, stage1_epochs: int = 10,
                    stage2_epochs: int = 20, batch_size: int = 8,
                    adapter_lr: float = 5e-4, encoder_lr: float = 2e-5,
                    head_lr: float = 1e-4, save_as: str = "") -> dict:
    """One D1 domain-fusion arm x domain x seed.

    Timeout is 6h rather than 4h: the auxiliary stage sees 1,681-2,189 sentences,
    two to four times the target's own training split.
    """
    import sys

    sys.path.insert(0, "/root/research")
    from aste_full_train import run_fusion

    payload = run_fusion(domain=domain, arm=arm, seed=seed, model_id=model_id,
                         stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs,
                         batch_size=batch_size, adapter_lr=adapter_lr,
                         encoder_lr=encoder_lr, head_lr=head_lr,
                         data_root="/root/Data/ASTE")
    name = save_as or arm
    if save_as:
        payload["preset"] = save_as
    out = Path(f"/results/aste/{domain}/{name}/seed_{seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + chr(10),
        encoding="utf-8",
    )
    results_vol.commit()
    t = payload["test"]
    s1 = payload["stages"].get("stage1", {})
    print(f"[{name}/{domain}/seed{seed}] triplet P={t['precision']:.4f} "
          f"R={t['recall']:.4f} F1={t['f1']:.4f} (dev {payload['dev_f1']:.4f}) "
          f"aux={payload['aux_stats'].get('kept', 0)} stage1_dev={s1.get('dev_f1')} "
          f"adapters={payload['params']['adapters']:,} "
          f"trainable={payload['params']['trainable']:,}", flush=True)
    return payload


@app.function(image=image, gpu=GPU_TYPE, cpu=8, memory=32768, timeout=7200,
              volumes={"/results": results_vol, "/root/.cache/huggingface": hf_vol})
def aste_speed(encoder_ids: list[str] | None = None, batch_sizes: list[int] | None = None,
               reladapt_only: bool = False, emit_all: bool = False) -> dict:
    """Inference cost of RelAdapt versus T5-base generation on the four test sets."""
    import sys

    sys.path.insert(0, "/root/research")
    from aste_speed import benchmark

    report = benchmark("/root/Data/ASTE",
                       encoder_ids=tuple(encoder_ids or ("fastino/gliner2-base-v1",)),
                       batch_sizes=tuple(batch_sizes or (1, 32)),
                       reladapt_only=reladapt_only, emit_all=emit_all)
    out = Path("/results/speed")
    out.mkdir(parents=True, exist_ok=True)
    (out / ("speed_emit_all.json" if emit_all else "speed.json")).write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    results_vol.commit()
    return report


@app.function(image=image, timeout=600)
def aste_data_overlap() -> dict:
    """Cross-domain sentence overlap and the resulting clean auxiliary pools.

    Run this before quoting any pooled-data number: 16res.train+dev contains
    100% of 15res.test, so an unfiltered union trains on its own test set.
    """
    import sys

    sys.path.insert(0, "/root/research")
    from aste_data import overlap_report

    report = overlap_report("/root/Data/ASTE")
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.function(image=image, gpu=GPU_TYPE, cpu=4, timeout=3600,
              volumes={"/results": results_vol, "/root/.cache/huggingface": hf_vol})
def aste_zeroshot_probe(domain: str = "15res", model_id: str = BASE_MODEL,
                        aspect_label: str = "target",
                        opinion_label: str = "opinion word") -> dict:
    """Zero-shot pruning recall of GLiNER2's own span/label scorer.

    Nothing is trained. The question is whether the pretrained scorer reaches the
    gold spans that the fine-tuned model misses on unseen vocabulary, which is
    where B2h loses 13.6-26.6 points of recall.
    """
    import sys

    sys.path.insert(0, "/root/research")
    from aste_zeroshot import probe

    report = probe(domain=domain, model_id=model_id, data_root="/root/Data/ASTE",
                   aspect_label=aspect_label, opinion_label=opinion_label)
    out = Path(f"/results/zeroshot/{domain}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    results_vol.commit()
    return report


@app.function(image=image, gpu=GPU_TYPE, cpu=4, timeout=1800,
              volumes={"/root/.cache/huggingface": hf_vol})
def aste_align_debug(domain: str = "15res", n: int = 4) -> dict:
    """Dump the two tokenisations for sentences the zero-shot probe had to skip."""
    import sys

    sys.path.insert(0, "/root/research")
    from aste_zeroshot import debug_alignment

    return debug_alignment(domain=domain, data_root="/root/Data/ASTE", n=n)
