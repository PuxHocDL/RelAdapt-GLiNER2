"""Inference cost: one-pass schema-driven matching versus autoregressive generation.

Both sides process the same test sentences on the same GPU.

* RelAdapt: tokenisation and span enumeration (the collator), one encoder pass,
  role and relation matching, and set decoding - the full path to triplets.
  The matching layers are untrained and emit almost no pairs, so ``emit_all``
  additionally times the worst case in which all N x N = 100 candidate pairs of
  every sentence are decoded into triplets (a trained model emits about two).
* Generation (T5-base): greedy decoding, with the number of generated tokens
  forced to the length of the gold target string in that paradigm's format
  (``min_new_tokens == max_new_tokens``), batch-wise to the longest target, as a
  real batch keeps decoding until its longest member stops. Speed depends on
  model size and output length, not on the trained weights, so an untrained T5
  gives the decoding cost of a trained one. Constrained decoding, string parsing
  and repair are not timed, so this is a lower bound on the generative cost.

  GAS   extraction-style target "(aspect, opinion, positive); (...)", one pass.
  MvP   five element-order views "[A] a [O] o [S] great [SSEP] ...", five passes,
        each with its order prompt appended to the input, as in Gou et al. (2023).
"""

from __future__ import annotations

import itertools
import time

import torch
from torch.utils.data import DataLoader

from aste_data import load_split
from aste_full import AsteConfig, AsteModel
from aste_full_train import AsteCollator, decode, encode

POL_GAS = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}
POL_MVP = {"POS": "great", "NEU": "ok", "NEG": "bad"}
MVP_VIEWS = list(itertools.permutations("AOS"))[:5]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _targets(examples):
    gas, mvp = [], {v: [] for v in MVP_VIEWS}
    for ex in examples:
        tri = [(ex.text[t.aspect[0]:t.aspect[1]], ex.text[t.opinion[0]:t.opinion[1]], t.polarity)
               for t in ex.triplets]
        gas.append("; ".join(f"({a}, {o}, {POL_GAS[p]})" for a, o, p in tri))
        for v in MVP_VIEWS:
            el = lambda a, o, p, k: {"A": f"[A] {a}", "O": f"[O] {o}", "S": f"[S] {POL_MVP[p]}"}[k]  # noqa: E731
            mvp[v].append(" [SSEP] ".join(" ".join(el(a, o, p, k) for k in v) for a, o, p in tri))
    return gas, mvp


def _time_reladapt(model_id, examples, batch_size, device, warmup=16, emit_all=False):
    if model_id.startswith("fastino/"):
        from gliner2 import AutoExtractor
        bb = AutoExtractor.from_pretrained(model_id)
        encoder, tokenizer = bb.encoder, bb.processor.tokenizer
    else:
        from transformers import AutoModel, AutoTokenizer
        encoder = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    cfg = AsteConfig.preset("R1bm")
    cfg.hidden_size = encoder.config.hidden_size
    model = AsteModel(encoder, cfg).to(device).eval()
    collate = AsteCollator(tokenizer.pad_token_id, cfg)

    def run(exs):
        # Tokenisation is inside the timed region, as it is for generation.
        rows, _ = encode(exs, tokenizer, 128)
        n_pred = 0
        for batch in DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate):
            t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(t["input_ids"], t["attention_mask"], t["spans"], t["span_valid"])
            if emit_all:
                out["pair_logits"][..., 0] -= 1e4      # every candidate pair beats NONE
            n_pred += sum(len(p) for p in decode(out, batch["batch"], batch["span_key"]))
        return n_pred

    with torch.inference_mode():
        run(examples[:warmup])
        _sync()
        t0 = time.perf_counter()
        n_pred = run(examples)
        _sync()
        dt = time.perf_counter() - t0
    del model, encoder
    torch.cuda.empty_cache()
    return dt, {"emitted_pairs": n_pred}


def _time_generation(model, tok, inputs, targets, batch_size, device, warmup=16):
    tgt_len = [len(tok(t).input_ids) for t in targets]      # includes </s>

    def run(lo, hi):
        gen = 0
        for i in range(lo, hi, batch_size):
            j = min(i + batch_size, hi)
            enc = tok(inputs[i:j], return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(device)
            n = max(tgt_len[i:j])
            model.generate(**enc, do_sample=False, num_beams=1,
                           min_new_tokens=n, max_new_tokens=n)
            gen += n * (j - i)
        return gen

    with torch.inference_mode():
        run(0, min(warmup, len(inputs)))
        _sync()
        t0 = time.perf_counter()
        gen = run(0, len(inputs))
        _sync()
        dt = time.perf_counter() - t0
    return dt, {"mean_target_tokens": sum(tgt_len) / len(tgt_len), "generated_tokens": gen}


def benchmark(data_root: str, domains=("14lap", "14res", "15res", "16res"),
              encoder_ids=("fastino/gliner2-base-v1",), t5_id: str = "google-t5/t5-base",
              batch_sizes=(1, 32), reladapt_only: bool = False, emit_all: bool = False) -> dict:
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    examples = [ex for d in domains for ex in load_split(d, "test", data_root)]
    n = len(examples)
    gas_t, mvp_t = _targets(examples)
    texts = [ex.text for ex in examples]

    rows = []

    def add(system, bs, dt, extra):
        rows.append({"system": system, "batch_size": bs, "seconds": round(dt, 3),
                     "sent_per_s": round(n / dt, 1), "ms_per_sent": round(1000 * dt / n, 2),
                     **extra})
        print(rows[-1], flush=True)

    for bs in batch_sizes:
        for enc_id in encoder_ids:
            dt, extra = _time_reladapt(enc_id, examples, bs, device, emit_all=emit_all)
            add(f"RelAdapt [{enc_id}]" + (" emit-all" if emit_all else ""), bs, dt, extra)
    if reladapt_only:
        return {"n_sentences": n, "domains": list(domains), "emit_all": emit_all,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "rows": rows}

    tok = T5TokenizerFast.from_pretrained(t5_id)
    t5 = T5ForConditionalGeneration.from_pretrained(t5_id).to(device).eval()
    for bs in batch_sizes:
        dt, extra = _time_generation(t5, tok, texts, gas_t, bs, device)
        add("T5-base generation, GAS extraction style", bs, dt, extra)
        total, gen = 0.0, 0
        for v in MVP_VIEWS:
            prompt = " ".join(f"[{k}]" for k in v)
            dt, extra = _time_generation(t5, tok, [f"{x} {prompt}" for x in texts],
                                         mvp_t[v], bs, device)
            total += dt
            gen += extra["generated_tokens"]
        add("T5-base generation, MvP style (5 views)", bs, total,
            {"generated_tokens": gen, "views": len(MVP_VIEWS)})

    return {"n_sentences": n, "domains": list(domains), "t5": t5_id,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "rows": rows}
