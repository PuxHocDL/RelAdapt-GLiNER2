"""Train and evaluate full ASTE triplet extraction.

The metric is exact triplet F1: a prediction counts only when the aspect span,
the opinion span and the sentiment all match a gold triplet. Nothing is given at
inference - spans are enumerated and scored by the model itself - so this is the
setting the published ASTE numbers refer to.
"""

from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from aste_data import _char_to_token_span, load_split
from aste_full import POLARITY_TO_PAIR, AsteConfig, AsteModel, enumerate_spans


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(examples, tokenizer, max_length: int):
    """Tokenise and convert every gold triplet into token-span coordinates."""
    out, dropped = [], 0
    for ex in examples:
        enc = tokenizer(ex.text, return_offsets_mapping=True,
                        truncation=True, max_length=max_length)
        offs = enc["offset_mapping"]
        triplets = []
        for t in ex.triplets:
            a = _char_to_token_span(offs, *t.aspect)
            o = _char_to_token_span(offs, *t.opinion)
            if a is None or o is None:
                dropped += 1
                continue
            triplets.append((a, o, POLARITY_TO_PAIR[t.polarity]))
        out.append({
            "id": ex.id,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "triplets": triplets,
            "n_tokens": len(enc["input_ids"]),
        })
    return out, {"examples": len(out), "dropped_triplets": dropped}


def polarity_counts(rows) -> tuple[int, int, int]:
    """(POS, NEU, NEG) triplet counts in an encoded split.

    Read off the split the run actually trains on rather than hard-coded, so a
    fusion stage that sees the auxiliary pool gets the auxiliary pool's class
    balance - which is the whole reason that stage exists for NEU: 15res holds
    25 neutral triplets of its own against 343 in the clean auxiliary union.
    """
    c = [0, 0, 0]
    for r in rows:
        for _, _, lab in r["triplets"]:
            c[lab - 1] += 1
    return (c[0], c[1], c[2])


class AsteCollator:
    def __init__(self, pad_id: int, cfg: AsteConfig):
        self.pad_id = pad_id
        self.cfg = cfg

    def __call__(self, batch):
        bs = len(batch)
        max_tok = max(b["n_tokens"] for b in batch)
        input_ids = torch.full((bs, max_tok), self.pad_id, dtype=torch.long)
        attention = torch.zeros((bs, max_tok), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["n_tokens"]
            input_ids[i, :n] = torch.tensor(b["input_ids"])
            attention[i, :n] = torch.tensor(b["attention_mask"])

        spans = enumerate_spans(max_tok, self.cfg.max_span_width, torch.device("cpu"))
        span_key = {(int(s), int(e)): i for i, (s, e) in enumerate(spans)}
        n_spans = spans.shape[0]

        span_valid = torch.zeros((bs, n_spans), dtype=torch.bool)
        gold_a = torch.zeros((bs, n_spans), dtype=torch.bool)
        gold_o = torch.zeros((bs, n_spans), dtype=torch.bool)
        for i, b in enumerate(batch):
            span_valid[i] = spans[:, 1] <= b["n_tokens"]
            for a, o, _ in b["triplets"]:
                if a in span_key:
                    gold_a[i, span_key[a]] = True
                if o in span_key:
                    gold_o[i, span_key[o]] = True

        # Token-level targets for the auxiliary tagger: channel 0 = inside an
        # aspect span, channel 1 = inside an opinion span.
        token_target = torch.zeros((bs, max_tok, 2), dtype=torch.float)
        for i, b in enumerate(batch):
            for (a0, a1), (o0, o1), _ in b["triplets"]:
                token_target[i, a0:a1, 0] = 1.0
                token_target[i, o0:o1, 1] = 1.0

        out = {
            "input_ids": input_ids, "attention_mask": attention,
            "spans": spans, "span_valid": span_valid,
            "gold_aspects": gold_a, "gold_opinions": gold_o,
            "token_target": token_target, "token_mask": attention.bool(),
            "batch": batch, "span_key": span_key,
        }
        # The frozen GLiNER2 prior, if it was attached. Spans the prior has no
        # entry for stay at 0, which is silence rather than evidence against:
        # the model adds `schema_gate * score` to the span logit.
        if batch[0].get("schema") is not None:
            schema = torch.zeros((bs, n_spans, 2), dtype=torch.float)
            for i, b in enumerate(batch):
                for key, (a_sc, o_sc) in b["schema"].items():
                    j = span_key.get(key)
                    if j is not None:
                        schema[i, j, 0] = a_sc
                        schema[i, j, 1] = o_sc
            out["schema_scores"] = schema
        return out


def build_pair_target(out, batch_meta, span_key, device):
    """(B,Na,No) label per kept pair: none / POS / NEU / NEG."""
    a_idx, o_idx = out["aspect_idx"], out["opinion_idx"]
    bs, na = a_idx.shape
    no = o_idx.shape[1]
    target = torch.zeros((bs, na, no), dtype=torch.long, device=device)
    for i, b in enumerate(batch_meta):
        pos_a = {int(a_idx[i, j]): j for j in range(na)}
        pos_o = {int(o_idx[i, j]): j for j in range(no)}
        for a, o, label in b["triplets"]:
            ai, oi = span_key.get(a), span_key.get(o)
            if ai in pos_a and oi in pos_o:
                target[i, pos_a[ai], pos_o[oi]] = label
    return target


def build_coupled_targets(out, batch_meta, span_key):
    """Gold triplets expressed in the model's selected aspect/opinion slots."""
    a_idx, o_idx = out["aspect_idx"], out["opinion_idx"]
    bs, na = a_idx.shape
    no = o_idx.shape[1]
    per_example = []
    for i, b in enumerate(batch_meta):
        slot_a = {int(a_idx[i, j]): j for j in range(na)}
        slot_o = {int(o_idx[i, j]): j for j in range(no)}
        pairs = []
        for a, o, label in b["triplets"]:
            ai, oi = span_key.get(a), span_key.get(o)
            if ai in slot_a and oi in slot_o:
                pairs.append((slot_a[ai], slot_o[oi], label))
        per_example.append(pairs)
    return per_example


@torch.no_grad()
def decode_coupled(out, batch_meta, span_key):
    """A pair is emitted when it beats the dustbin from both sides.

    No threshold: the assignment itself decides existence, which is the point of
    merging the "none" class into the dustbin.
    """
    index_to_span = {v: k for k, v in span_key.items()}
    log_p = out["log_assign"]
    b, na1, no1 = log_p.shape
    na, no = na1 - 1, no1 - 1
    core = log_p[:, :na, :no]
    a_bin = log_p[:, :na, no].unsqueeze(-1)      # this aspect stays unmatched
    o_bin = log_p[:, na, :no].unsqueeze(1)       # this opinion stays unmatched
    keep = (core > a_bin) & (core > o_bin)
    pol = out["polarity_logits"].argmax(-1) + 1
    a_idx, o_idx = out["aspect_idx"], out["opinion_idx"]

    preds = []
    for i in range(len(batch_meta)):
        found = set()
        for j, k in zip(*torch.nonzero(keep[i], as_tuple=True)):
            found.add((index_to_span[int(a_idx[i, j])],
                       index_to_span[int(o_idx[i, k])], int(pol[i, j, k])))
        preds.append(found)
    return preds


@torch.no_grad()
def decode(out, batch_meta, span_key, threshold: float = 0.0,
           verifier_threshold: float | None = 0.5):
    if out.get("coupled"):
        return decode_coupled(out, batch_meta, span_key)
    """Predicted triplets per example, as {(aspect_span, opinion_span, label)}."""
    index_to_span = {v: k for k, v in span_key.items()}
    probs = out["pair_logits"].softmax(-1)
    best = probs[..., 1:].argmax(-1) + 1
    conf = probs[..., 1:].max(-1).values
    none_p = probs[..., 0]
    a_idx, o_idx = out["aspect_idx"], out["opinion_idx"]
    # The verifier can only remove a pair the generator already wanted to emit.
    vkeep = None
    if out.get("verifier_logits") is not None and verifier_threshold is not None:
        vkeep = out["verifier_logits"].sigmoid() >= verifier_threshold

    preds = []
    for i in range(len(batch_meta)):
        found = set()
        keep = (conf[i] > none_p[i]) & (conf[i] >= threshold)
        if vkeep is not None:
            keep = keep & vkeep[i]
        for j, k in zip(*torch.nonzero(keep, as_tuple=True)):
            a_span = index_to_span[int(a_idx[i, j])]
            o_span = index_to_span[int(o_idx[i, k])]
            found.add((a_span, o_span, int(best[i, j, k])))
        preds.append(found)
    return preds


def triplet_prf(golds, preds):
    tp = sum(len(g & p) for g, p in zip(golds, preds))
    n_pred = sum(len(p) for p in preds)
    n_gold = sum(len(g) for g in golds)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.0, keep_predictions=False):
    model.eval()
    golds, preds, records = [], [], []
    # Pruning recall: the share of gold spans that survive top-k selection at
    # all. Everything downstream - pairing, polarity, the assignment - is capped
    # by it, so a triplet-F1 change with flat pruning recall came from somewhere
    # else. No gold injection happens here, unlike in training.
    prune = {"gold": 0, "aspect": 0, "opinion": 0, "both": 0}
    # Where a gold triplet is lost, in one bucket each so the buckets sum to
    # n_gold. Every intervention this project has tried so far aimed at the span
    # pathway; this says how much of the loss is actually there.
    fate = ({"hit": 0, "span_missed": 0, "rejected_as_none": 0, "wrong_polarity": 0}
            if keep_predictions else None)
    # And the mirror question, on the prediction side. Every false positive is
    # labelled by how each of its two spans relates to the gold spans of its own
    # role: exact, overlapping but off the boundary, or unrelated. `polarity`
    # counts a pair that is a real gold pair carrying the wrong sentiment, which
    # is the same event `wrong_polarity` counts from the gold side.
    fp = ({"polarity": 0} if keep_predictions else None)
    # The mechanism metric for the K-series, recorded on every arm so a control
    # carries it too: the share of gold spans whose score beats every overlapping
    # rival. Unlike triplet F1 it does not depend on pairing, on polarity or on
    # how many candidates survive pruning, so it isolates the span competition
    # that the snap-boundary oracle says is worth +6.90 F1. No gold is injected
    # into the forward pass - this reads the logits the model produces on its own.
    rank = {"aspect_wins": 0, "aspect_n": 0, "opinion_wins": 0, "opinion_n": 0,
            "aspect_wins_1edge": 0, "opinion_wins_1edge": 0}
    funnel = ({"n_gold": 0, "not_enumerable": 0, "pruned_lost_to_rival": 0,
               "pruned_no_rival": 0, "survived_pair_rejected": 0,
               "survived_wrong_polarity": 0, "hit": 0} if keep_predictions else None)
    # Counts for the two existence oracles and the margin/NONE-competition split.
    exist_counts = {"tp": 0, "n_pred": 0, "tp_pol": 0, "n_pred_pol": 0}
    exist_diag = {}
    for batch in loader:
        t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(t["input_ids"], t["attention_mask"], t["spans"], t["span_valid"],
                    schema_scores=t.get("schema_scores"))
        if not out.get("coupled"):
            from aste_ranking import overlap_rank_accuracy
            for role, logit_key, gold_key in (
                    ("aspect", "aspect_logits", "gold_aspects"),
                    ("opinion", "opinion_logits", "gold_opinions")):
                for tag, one_edge in (("", False), ("_1edge", True)):
                    w, n = overlap_rank_accuracy(
                        out[logit_key], t[gold_key], t["spans"], t["span_valid"],
                        single_edge_only=one_edge)
                    rank[f"{role}_wins{tag}"] += w
                    if not tag:
                        rank[f"{role}_n"] += n
        batch_preds = decode(out, batch["batch"], batch["span_key"], threshold)
        index_to_span = {v: k for k, v in batch["span_key"].items()}
        a_idx, o_idx = out["aspect_idx"], out["opinion_idx"]
        # Decoder funnel: for each gold triplet, the FIRST stage that loses it.
        # The stages are the decode path in order, so the buckets are disjoint and
        # sum to n_gold. This is what separates "the span was never a candidate"
        # from "the span was a candidate and lost to an overlapping rival" from
        # "the span won and the pair was still rejected" - three failures that a
        # single span-level metric cannot tell apart, and which need different
        # fixes.
        if funnel is not None and not out.get("coupled"):
            probs_f = out["pair_logits"].softmax(-1)
            forced = [set() for _ in batch["batch"]]
            forced_pol = [set() for _ in batch["batch"]]
            from aste_relation import existence_diagnostics
            tgt_f = build_pair_target(out, batch["batch"], batch["span_key"], device)
            for k, v in existence_diagnostics(out["pair_logits"], tgt_f).items():
                exist_diag[k] = exist_diag.get(k, 0) + v
            for i, b in enumerate(batch["batch"]):
                kept_a = {index_to_span[int(x)]: j for j, x in enumerate(a_idx[i])}
                kept_o = {index_to_span[int(x)]: j for j, x in enumerate(o_idx[i])}
                for a, o, lab in b["triplets"]:
                    funnel["n_gold"] += 1
                    if a not in batch["span_key"] or o not in batch["span_key"]:
                        funnel["not_enumerable"] += 1
                        continue
                    if a not in kept_a or o not in kept_o:
                        # Did an overlapping rival of the missing span survive in
                        # its place? That is a lost race, not a missing candidate.
                        lost = False
                        for span, kept in ((a, kept_a), (o, kept_o)):
                            if span in kept:
                                continue
                            if any(not (span[1] <= c[0] or c[1] <= span[0]) for c in kept):
                                lost = True
                        funnel["pruned_lost_to_rival" if lost else "pruned_no_rival"] += 1
                        continue
                    cell = probs_f[i, kept_a[a], kept_o[o]]
                    best = int(cell[1:].argmax()) + 1
                    conf = float(cell[1:].max())
                    if not (conf > float(cell[0]) and conf >= threshold):
                        funnel["survived_pair_rejected"] += 1
                        # Existence oracles. The pair is rejected but both spans
                        # are alive, so it is recoverable without touching span
                        # selection: force it out, once carrying the polarity the
                        # model already prefers and once carrying the gold one.
                        # 16.94% of gold triplets dying here is not 16.94 F1 of
                        # headroom - forcing them costs precision - and this is
                        # what measures the difference.
                        forced[i].add((a, o, best))
                        forced_pol[i].add((a, o, lab))
                    elif best != lab:
                        funnel["survived_wrong_polarity"] += 1
                    else:
                        funnel["hit"] += 1
            # Fold the forced relations into the normal prediction set and score
            # it the ordinary way, so the oracle pays the precision cost of every
            # relation it adds. n_pred grows; that is the point.
            for i, b in enumerate(batch["batch"]):
                gold_set = {(a, o, lab) for a, o, lab in b["triplets"]}
                base_pred = batch_preds[i]
                for key, extra in (("", forced[i]), ("_pol", forced_pol[i])):
                    merged = set(base_pred) | extra
                    exist_counts["tp" + key] += len(gold_set & merged)
                    exist_counts["n_pred" + key] += len(merged)
        probs = (out["pair_logits"].softmax(-1) if fate is not None
                 and not out.get("coupled") else None)
        for i, b in enumerate(batch["batch"]):
            slot_a = {index_to_span[int(x)]: j for j, x in enumerate(a_idx[i])}
            slot_o = {index_to_span[int(x)]: j for j, x in enumerate(o_idx[i])}
            cand_a, cand_o = set(slot_a), set(slot_o)
            for a, o, lab in b["triplets"]:
                prune["gold"] += 1
                prune["aspect"] += int(a in cand_a)
                prune["opinion"] += int(o in cand_o)
                both = a in cand_a and o in cand_o
                prune["both"] += int(both)
                if probs is None:
                    continue
                if not both:
                    fate["span_missed"] += 1
                    continue
                cell = probs[i, slot_a[a], slot_o[o]]
                best = int(cell[1:].argmax()) + 1
                conf = float(cell[1:].max())
                if not (conf > float(cell[0]) and conf >= threshold):
                    fate["rejected_as_none"] += 1
                elif best != lab:
                    fate["wrong_polarity"] += 1
                else:
                    fate["hit"] += 1
        if fp is not None:
            for b, pred in zip(batch["batch"], batch_preds):
                gold = {(a, o, lab) for a, o, lab in b["triplets"]}
                gold_pairs = {(a, o) for a, o, _ in b["triplets"]}
                g_asp = {a for a, _, _ in b["triplets"]}
                g_opi = {o for _, o, _ in b["triplets"]}

                def status(span, pool):
                    if span in pool:
                        return "exact"
                    lo, hi = span
                    if any(lo < b_ and a_ < hi for a_, b_ in pool):
                        return "overlap"
                    return "none"

                for a, o, lab in pred - gold:
                    if (a, o) in gold_pairs:
                        fp["polarity"] += 1
                        continue
                    key = f"a:{status(a, g_asp)}/o:{status(o, g_opi)}"
                    fp[key] = fp.get(key, 0) + 1
        for b, p in zip(batch["batch"], batch_preds):
            g = {(a, o, lab) for a, o, lab in b["triplets"]}
            golds.append(g)
            preds.append(p)
            if keep_predictions:
                records.append({"id": b["id"],
                                "gold": sorted(map(list, map(lambda x: (list(x[0]), list(x[1]), x[2]), g))),
                                "pred": sorted(map(list, map(lambda x: (list(x[0]), list(x[1]), x[2]), p))),
                                "tp": len(g & p), "n_gold": len(g), "n_pred": len(p)})
    metrics = triplet_prf(golds, preds)
    # Per-polarity triplet P/R/F1. Recorded on every run because the whole
    # H-series rests on one class: B2h predicts NEU 0 times on 15res and 5 times
    # on 16res against 145 gold, and no corpus-level F1 shows that.
    per_pol = {}
    for lab, name in ((1, "POS"), (2, "NEU"), (3, "NEG")):
        tp = sum(sum(1 for t in g & p if t[2] == lab) for g, p in zip(golds, preds))
        n_pred = sum(sum(1 for t in p if t[2] == lab) for p in preds)
        n_gold = sum(sum(1 for t in g if t[2] == lab) for g in golds)
        pr = tp / n_pred if n_pred else 0.0
        rc = tp / n_gold if n_gold else 0.0
        per_pol[name] = {"precision": round(pr, 4), "recall": round(rc, 4),
                         "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
                         "tp": tp, "n_pred": n_pred, "n_gold": n_gold}
    metrics["per_polarity"] = per_pol
    if funnel is not None and funnel["n_gold"]:
        metrics["funnel"] = funnel
        ng = funnel["n_gold"]

        def _prf(tp, npd):
            pr = tp / npd if npd else 0.0
            rc = tp / ng if ng else 0.0
            return {"precision": round(pr, 4), "recall": round(rc, 4),
                    "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}

        metrics["existence_oracle"] = {
            "force_existence": _prf(exist_counts["tp"], exist_counts["n_pred"]),
            "force_existence_and_polarity": _prf(exist_counts["tp_pol"],
                                                 exist_counts["n_pred_pol"]),
        }
        # Share of gold relations emitted given BOTH spans survived pruning -
        # the metric the candidate-set formulation is supposed to move.
        survived = (funnel["survived_pair_rejected"] + funnel["survived_wrong_polarity"]
                    + funnel["hit"])
        if survived:
            metrics["emit_recall_given_survive"] = round(
                (funnel["survived_wrong_polarity"] + funnel["hit"]) / survived, 4)
        if exist_diag:
            metrics["existence_diag"] = exist_diag
    if rank["aspect_n"]:
        metrics["overlap_rank"] = {
            "aspect": round(rank["aspect_wins"] / rank["aspect_n"], 4),
            "opinion": round(rank["opinion_wins"] / max(rank["opinion_n"], 1), 4),
            "aspect_1edge": round(rank["aspect_wins_1edge"] / rank["aspect_n"], 4),
            "opinion_1edge": round(rank["opinion_wins_1edge"] / max(rank["opinion_n"], 1), 4),
            "n_aspect": rank["aspect_n"], "n_opinion": rank["opinion_n"],
        }
    if fate is not None:
        metrics["fate"] = {k: v for k, v in fate.items()}
        metrics["fp_breakdown"] = dict(sorted(fp.items(), key=lambda kv: -kv[1]))
    n = max(1, prune["gold"])
    metrics["prune_recall"] = {"aspect": round(prune["aspect"] / n, 4),
                               "opinion": round(prune["opinion"] / n, 4),
                               "both": round(prune["both"] / n, 4),
                               "n_gold_spans": prune["gold"]}
    if keep_predictions:
        metrics["predictions"] = records
    return metrics


@torch.no_grad()
def decode_margin(out, batch_meta, span_key, tau: float):
    """Emit when the best polarity logit beats the NONE logit by more than tau.

    The default decode is this rule at tau = 0. Sweeping tau on dev separates how
    much of the 16.94% NONE-rejection bucket is a calibration offset from how
    much needs a different objective - if a single global tau recovers most of
    it, no new loss is warranted.
    """
    index_to_span = {v: k for k, v in span_key.items()}
    logits = out["pair_logits"]
    margin = logits[..., 1:].max(-1).values - logits[..., 0]
    best = logits[..., 1:].argmax(-1) + 1
    a_idx, o_idx = out["aspect_idx"], out["opinion_idx"]
    preds = []
    for i in range(len(batch_meta)):
        found = set()
        for j, k in zip(*torch.nonzero(margin[i] > tau, as_tuple=True)):
            found.add((index_to_span[int(a_idx[i, j])],
                       index_to_span[int(o_idx[i, k])], int(best[i, j, k])))
        preds.append(found)
    return preds


@torch.no_grad()
def evaluate_verifier_sweep(model, loader, device, thresholds):
    """P/R/F1 and the false-positive type mix at several verifier thresholds.

    One encoder pass, many decodes. The `fp_one_span_real` column is the one the
    verifier was built to move, and reading it alongside precision and recall is
    what separates "good ranking, wrong calibration" - a stretch of thresholds
    where precision climbs while recall holds - from "same trade, better
    exchange rate", where the two move against each other all the way along.
    """
    model.eval()
    golds, preds = [], {t: [] for t in thresholds}
    for batch in loader:
        t_in = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(t_in["input_ids"], t_in["attention_mask"],
                    t_in["spans"], t_in["span_valid"],
                    schema_scores=t_in.get("schema_scores"))
        if out.get("verifier_logits") is None:
            return {}
        for b in batch["batch"]:
            golds.append({(a, o, lab) for a, o, lab in b["triplets"]})
        for th in thresholds:
            preds[th].extend(decode(out, batch["batch"], batch["span_key"],
                                    0.0, verifier_threshold=th))

    def overlaps(x, y):
        return not (x[1] <= y[0] or y[1] <= x[0])

    out_all = {}
    for th in thresholds:
        m = triplet_prf(golds, preds[th])
        fp = {"one_span_real": 0, "wrong_partner": 0, "wrong_polarity": 0,
              "boundary": 0, "spurious": 0}
        for g, p in zip(golds, preds[th]):
            g_asp = {x for x, _, _ in g}
            g_opi = {y for _, y, _ in g}
            pairs = {(x, y) for x, y, _ in g}
            for a, o, lab in p - g:
                if (a, o) in pairs:
                    fp["wrong_polarity"] += 1
                elif a in g_asp and o in g_opi:
                    fp["wrong_partner"] += 1
                elif a in g_asp or o in g_opi:
                    fp["one_span_real"] += 1
                elif any(overlaps(a, x) for x in g_asp) or any(overlaps(o, y) for y in g_opi):
                    fp["boundary"] += 1
                else:
                    fp["spurious"] += 1
        m["fp_types"] = fp
        out_all[th] = m
    return out_all


@torch.no_grad()
def evaluate_margin_sweep(model, loader, device, taus):
    """P/R/F1 at several NULL margins from one encoder pass."""
    model.eval()
    golds, preds = [], {t: [] for t in taus}
    for batch in loader:
        t_in = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(t_in["input_ids"], t_in["attention_mask"],
                    t_in["spans"], t_in["span_valid"],
                    schema_scores=t_in.get("schema_scores"))
        if out.get("coupled"):
            return {}
        for b in batch["batch"]:
            golds.append({(a, o, lab) for a, o, lab in b["triplets"]})
        for tau in taus:
            preds[tau].extend(decode_margin(out, batch["batch"], batch["span_key"], tau))
    return {tau: triplet_prf(golds, preds[tau]) for tau in taus}


@torch.no_grad()
def evaluate_sweep(model, loader, device, thresholds):
    """Metrics at several decision thresholds from a single forward pass.

    Our recall already matches or beats published systems while precision trails
    them by 10-14 points, so the question is how much of that gap is just an
    uncalibrated cut-off. One encoder pass, many decodes: the sweep is close to
    free next to training.
    """
    model.eval()
    golds, preds = [], {t: [] for t in thresholds}
    for batch in loader:
        t_in = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(t_in["input_ids"], t_in["attention_mask"],
                    t_in["spans"], t_in["span_valid"],
                    schema_scores=t_in.get("schema_scores"))
        for b in batch["batch"]:
            golds.append({(a, o, lab) for a, o, lab in b["triplets"]})
        for th in thresholds:
            preds[th].extend(decode(out, batch["batch"], batch["span_key"], th))
    return {th: triplet_prf(golds, preds[th]) for th in thresholds}


def encoder_lr_groups(encoder, base_lr: float, decay: float):
    """One parameter group per encoder layer, LR decaying towards the embeddings.

    The collapses land right after the one-cycle peak, which points at the peak LR
    reaching the lower layers. Layer-wise decay keeps the top of the encoder fast
    while the layers that carry the general representation move slowly.
    """
    layers = None
    for attr in ("encoder", "transformer"):
        mod = getattr(encoder, attr, None)
        if mod is not None and hasattr(mod, "layer"):
            layers = mod.layer
            break
    if layers is None:
        layers = getattr(encoder, "layer", None)
    if layers is None:                       # unknown architecture: no decay
        return [{"params": list(encoder.parameters()), "lr": base_lr}]

    layer_ids = [{id(p) for p in layer.parameters()} for layer in layers]
    n = len(layer_ids)
    groups, seen = [], set()
    for i, layer in enumerate(layers):
        lr = base_lr * (decay ** (n - 1 - i))          # top layer keeps base_lr
        groups.append({"params": list(layer.parameters()), "lr": lr})
        seen |= layer_ids[i]
    rest = [p for p in encoder.parameters() if id(p) not in seen]
    if rest:                                          # embeddings and any pooler
        groups.insert(0, {"params": rest, "lr": base_lr * (decay ** n)})
    return groups


def run_one(domain: str, preset: str, seed: int,
            model_id: str = "fastino/gliner2-base-v1",
            epochs: int = 20, batch_size: int = 8,
            encoder_lr: float = 2e-5, head_lr: float = 1e-4,
            max_length: int = 128, data_root: str = "/root/Data/ASTE",
            warmup_pct: float = 0.1, llrd: float = 1.0,
            gate_lr: float = 0.0, verifier_epochs: int = 10,
            verifier_lr: float = 1e-3) -> dict:
    from gliner2 import AutoExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    cfg = AsteConfig.preset(preset)
    backbone = AutoExtractor.from_pretrained(model_id)
    encoder = backbone.encoder
    tokenizer = backbone.processor.tokenizer
    cfg.hidden_size = encoder.config.hidden_size
    model = AsteModel(encoder, cfg).to(device)

    data, stats = {}, {}
    schema_stats = None
    for split in ("train", "dev", "test"):
        examples = load_split(domain, split, data_root)
        data[split], stats[split] = encode(examples, tokenizer, max_length)
        if cfg.use_schema_score:
            # Scored by the backbone as downloaded, before a single optimiser
            # step, and never updated again: the prior is a fixed feature, not a
            # second head that could drift with the encoder it is meant to
            # correct. Costs one forward pass per split at startup.
            from aste_zeroshot import schema_lookups
            lookups, st = schema_lookups(
                backbone, examples, tokenizer, max_length, cfg.max_span_width, device)
            for row, lk in zip(data[split], lookups):
                row["schema"] = lk
            schema_stats = {**(schema_stats or {}), split: st}
    collate = AsteCollator(tokenizer.pad_token_id, cfg)
    loaders = {
        s: DataLoader(rows, batch_size=batch_size, shuffle=(s == "train"),
                      collate_fn=collate)
        for s, rows in data.items()
    }
    train_polarity = polarity_counts(data["train"])
    model.set_polarity_weights(train_polarity)

    enc_ids = {id(p) for p in model.encoder.parameters()}
    # Two scalars deciding how far the frozen prior may move a span logit. At the
    # head LR they crawled to 0.03-0.04 over a whole run against a prior in
    # [0,1], which is not a decision by the model so much as a speed limit, so
    # they get their own group when `gate_lr` is set.
    gate = ([model.schema_gate] if gate_lr and hasattr(model, "schema_gate") else [])
    gate_ids = {id(p) for p in gate}
    head = [p for p in model.parameters()
            if id(p) not in enc_ids and id(p) not in gate_ids]
    if llrd < 1.0:
        groups = encoder_lr_groups(model.encoder, encoder_lr, llrd)
    else:
        groups = [{"params": list(model.encoder.parameters()), "lr": encoder_lr}]
    groups.append({"params": head, "lr": head_lr})
    if gate:
        groups.append({"params": gate, "lr": gate_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = max(1, len(loaders["train"]) * epochs)
    # A longer warmup is the standard remedy for a large encoder that diverges at
    # the learning rate which gives its best result on more data.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps,
        pct_start=warmup_pct, anneal_strategy="cos",
    )

    best = {"f1": -1.0, "state": None, "epoch": -1}
    history = []
    # Fingerprint of the data order, taken from the real training loop rather
    # than from a probe iterator - creating an extra iterator would itself draw
    # the shuffle permutation and change training. Two arms on the same seed must
    # produce the same hash; if they do not, the RNG-state fix in
    # AsteModel.__init__ has been defeated again and the comparison is void.
    order_ids, batch_order_hash = [], None
    for epoch in range(epochs):
        model.train()
        # Progress runs 0 -> 1 across training, so the assignment starts soft and
        # ends at the temperature the trained presets have always used.
        temp = model.set_sinkhorn_temp(epoch / max(1, epochs - 1))
        losses = []
        for batch in loaders["train"]:
            if epoch == 0 and len(order_ids) < 10:
                order_ids.append("|".join(b["id"] for b in batch["batch"]))
            t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(t["input_ids"], t["attention_mask"], t["spans"], t["span_valid"],
                        t["gold_aspects"], t["gold_opinions"],
                        schema_scores=t.get("schema_scores"))
            if out.get("coupled"):
                pairs = build_coupled_targets(out, batch["batch"], batch["span_key"])
                loss = model.compute_loss_coupled(out, pairs, t["span_valid"])["total"]
            else:
                target = build_pair_target(out, batch["batch"], batch["span_key"], device)
                loss = model.compute_loss(out, t["gold_aspects"], t["gold_opinions"],
                                          target, t["span_valid"],
                                          t.get("token_target"), t.get("token_mask"))["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            losses.append(float(loss))

        dev = evaluate(model, loaders["dev"], device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "dev_f1": dev["f1"], "sinkhorn_temp": temp})
        if dev["f1"] > best["f1"]:
            best = {"f1": dev["f1"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    import hashlib
    batch_order_hash = hashlib.sha256("~".join(order_ids).encode()).hexdigest()[:16]

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if cfg.use_verifier:
        # Stage 2. Everything except the verifier is frozen and the base has
        # already been selected on dev, so the generator is bit-identical to the
        # R1bm run with the same seed and any difference is the verifier alone.
        from aste_verifier import verifier_loss, verifier_targets
        model.verifier_active = True
        for prm in model.parameters():
            prm.requires_grad_(False)
        for prm in model.verifier.parameters():
            prm.requires_grad_(True)
        v_opt = torch.optim.AdamW(model.verifier.parameters(), lr=verifier_lr,
                                  weight_decay=0.01)
        v_best = {"f1": -1.0, "state": None, "epoch": -1}
        for epoch in range(verifier_epochs):
            model.train()
            model.encoder.eval()            # keep frozen dropout/BN deterministic
            for batch in loaders["train"]:
                t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                out = model(t["input_ids"], t["attention_mask"], t["spans"],
                            t["span_valid"], t["gold_aspects"], t["gold_opinions"],
                            schema_scores=t.get("schema_scores"))
                target = build_pair_target(out, batch["batch"], batch["span_key"], device)
                ga = t["gold_aspects"].gather(1, out["aspect_idx"])
                go = t["gold_opinions"].gather(1, out["opinion_idx"])
                label, weight, _ = verifier_targets(target, ga, go,
                                                    cfg.verifier_easy_weight)
                loss = verifier_loss(out["verifier_logits"], label, weight)
                v_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.verifier.parameters(), 1.0)
                v_opt.step()
            dv = evaluate(model, loaders["dev"], device)
            history.append({"epoch": epochs + epoch, "stage": "verifier",
                            "dev_f1": dv["f1"]})
            if dv["f1"] > v_best["f1"]:
                v_best = {"f1": dv["f1"], "epoch": epoch,
                          "state": {k: v.detach().cpu().clone()
                                    for k, v in model.verifier.state_dict().items()}}
        if v_best["state"] is not None:
            model.verifier.load_state_dict(v_best["state"])
        verifier_report = {"epochs": verifier_epochs, "best_epoch": v_best["epoch"],
                           "dev_f1": v_best["f1"], "lr": verifier_lr}
    else:
        verifier_report = None

    # Dev predictions on the SAME selected checkpoint. Every error analysis that
    # informs a design decision has to be computed here, not on test: the test
    # split has already been read across dozens of arms, so using its error
    # structure to choose a mechanism makes the later test number adaptive.
    dev_eval = evaluate(model, loaders["dev"], device, keep_predictions=True)
    dev_predictions = dev_eval.pop("predictions", [])

    test = evaluate(model, loaders["test"], device, keep_predictions=True)
    predictions = test.pop("predictions", [])

    # The threshold is chosen on dev and only then applied to test, so the tuned
    # number stays honest. threshold 0.0 is the untuned decode reported above.
    grid = [round(0.05 * i, 2) for i in range(0, 19)]
    dev_curve = evaluate_sweep(model, loaders["dev"], device, grid)
    best_th = max(grid, key=lambda t: dev_curve[t]["f1"])
    test_curve = evaluate_sweep(model, loaders["test"], device, grid)

    # 0.0 keeps every pair the generator emitted, so F1(t) - F1(0.0) is the
    # verifier's effect measured inside one run, with no trajectory confound.
    vthr = [round(0.1 * i, 1) for i in range(0, 10)]
    dev_verifier = (evaluate_verifier_sweep(model, loaders["dev"], device, vthr)
                    if cfg.use_verifier else {})

    taus = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    dev_margin = evaluate_margin_sweep(model, loaders["dev"], device, taus)
    best_tau = max(taus, key=lambda t: dev_margin[t]["f1"]) if dev_margin else 0.0
    test_margin = evaluate_margin_sweep(model, loaders["test"], device, taus)

    return {
        "task": "aste_triplet", "domain": domain, "preset": preset, "seed": seed,
        "model_id": model_id, "config": asdict(cfg),
        "params": model.added_parameter_count(),
        "epochs": epochs, "best_epoch": best["epoch"], "dev_f1": best["f1"],
        "encoder_lr": encoder_lr, "head_lr": head_lr, "warmup_pct": warmup_pct,
        "llrd": llrd, "gate_lr": gate_lr, "n_param_groups": len(groups),
        "test": test, "predictions": predictions, "history": history,
        "dev": dev_eval, "dev_predictions": dev_predictions,
        "tuned_threshold": best_th,
        "test_tuned": test_curve[best_th],
        "dev_curve": {str(t): dev_curve[t] for t in grid},
        "test_curve": {str(t): test_curve[t] for t in grid},
        "verifier_curve": {str(t): dev_verifier.get(t) for t in vthr} if dev_verifier else None,
        "null_margin": {"best_tau_on_dev": best_tau,
                        "dev": {str(t): dev_margin.get(t) for t in taus},
                        "test": {str(t): test_margin.get(t) for t in taus}},
        "data_stats": stats,
        "batch_order_hash": batch_order_hash,
        "train_polarity": {"POS": train_polarity[0], "NEU": train_polarity[1],
                           "NEG": train_polarity[2]},
        "polarity_weight": ([round(float(v), 4) for v in model.polarity_weight]
                            if hasattr(model, "polarity_weight") else None),
        "verifier": verifier_report,
        "dustbin_gate": (float(model.dustbin_gate.detach())
                         if hasattr(model, "dustbin_gate") else None),
        "schema_gate": ([round(float(v), 5) for v in model.schema_gate.detach()]
                        if hasattr(model, "schema_gate") else None),
        "schema_stats": schema_stats,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# ---------------------------------------------------------------------------
# D1: domain-fusion adapters
# ---------------------------------------------------------------------------

# Each arm is one hypothesis about where a gain would come from, and the set is
# built so that no arm can claim a win the next one down does not rule out.
#
#   F0  adapters, target data only        -> does a frozen backbone even hold up?
#   F1  adapters, auxiliary pre-stage     -> the method
#   F3  F1 with the shared adapter still  -> is freezing the shared adapter after
#       trainable at a low LR in stage 2     stage 1 helping or costing?
#   F2  no adapters, same two stages,     -> the control that matters: if plain
#       plain full fine-tuning               sequential transfer reaches the same
#                                            number, the adapter adds nothing
#
# The FH arms are the same schedule carrying the H3 model. They exist because
# the scarcest supervision in this corpus is scarce *per domain*, not in total:
# 15res holds 25 neutral triplets against 343 in its clean auxiliary pool
# (16res 50 vs 343), so the auxiliary stage is the only place where a neutral
# class can be learned at all. The sentiment prototypes are buffers and travel
# in the state dict, so stage 1's sentiment geometry is what stage 2 starts from.
#
#   FH0 H3 + adapters, target data only   -> the no-aux control for FH1
#   FH1 H3 + adapters + auxiliary stage   -> the method
#   FH2 H3, no adapters, same two stages  -> data or adapter?
FUSION_ARMS = {
    "F0": dict(preset="F",  use_aux=False, train_shared_stage2=True),
    "F1": dict(preset="F",  use_aux=True,  train_shared_stage2=False),
    "F3": dict(preset="F",  use_aux=True,  train_shared_stage2=True),
    "F2": dict(preset="F2", use_aux=True,  train_shared_stage2=True),
    "FH0": dict(preset="FH",  use_aux=False, train_shared_stage2=True),
    "FH1": dict(preset="FH",  use_aux=True,  train_shared_stage2=False),
    "FH3": dict(preset="FH",  use_aux=True,  train_shared_stage2=True),
    "FH2": dict(preset="FH2", use_aux=True,  train_shared_stage2=True),
}


def _only_trainable(model, groups) -> None:
    """requires_grad on exactly the parameters this stage optimises.

    Excluding a parameter from the optimizer already stops it being updated, but
    it would still accumulate gradient that nothing ever zeroes - in F1 the shared
    adapter is meant to be inert during stage 2, so it is switched off properly.
    """
    keep = {id(p) for g in groups for p in g["params"]}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)


def _fit(model, train_loader, dev_loader, groups, epochs, warmup_pct, device, tag):
    """One training stage. Returns the best-on-dev state and the epoch history.

    Deliberately a copy of ``run_one``'s loop rather than a refactor of it: the
    B2h numbers this arm is compared against were produced by that loop, and a
    silent change to the control is not worth the tidiness.
    """
    if epochs <= 0 or not groups:
        return None, []
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = max(1, len(train_loader) * epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps,
        pct_start=warmup_pct, anneal_strategy="cos",
    )
    best = {"f1": -1.0, "state": None, "epoch": -1}
    history = []
    for epoch in range(epochs):
        model.train()
        temp = model.set_sinkhorn_temp(epoch / max(1, epochs - 1))
        losses = []
        for batch in train_loader:
            t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(t["input_ids"], t["attention_mask"], t["spans"], t["span_valid"],
                        t["gold_aspects"], t["gold_opinions"],
                        schema_scores=t.get("schema_scores"))
            target = build_pair_target(out, batch["batch"], batch["span_key"], device)
            loss = model.compute_loss(out, t["gold_aspects"], t["gold_opinions"],
                                      target, t["span_valid"],
                                      t.get("token_target"), t.get("token_mask"))["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            sched.step()
            losses.append(float(loss))

        dev = evaluate(model, dev_loader, device)
        history.append({"stage": tag, "epoch": epoch, "loss": float(np.mean(losses)),
                        "dev_f1": dev["f1"], "sinkhorn_temp": temp})
        if dev["f1"] > best["f1"]:
            best = {"f1": dev["f1"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
    return best, history


def run_fusion(domain: str, arm: str, seed: int,
               model_id: str = "fastino/gliner2-base-v1",
               stage1_epochs: int = 10, stage2_epochs: int = 20,
               batch_size: int = 8, encoder_lr: float = 2e-5,
               adapter_lr: float = 5e-4, head_lr: float = 1e-4,
               max_length: int = 128, data_root: str = "/root/Data/ASTE",
               warmup_pct: float = 0.1) -> dict:
    """Two-stage domain-fusion training for one arm x domain x seed.

    Stage 1 trains on the leakage-filtered union of the other three domains,
    stage 2 on the target domain alone. Both stages select on the *target* dev
    set, which ``load_auxiliary`` keeps out of both training corpora, and the
    test set is touched once at the end.
    """
    from gliner2 import AutoExtractor

    from aste_adapter import (adapter_parameters, backbone_parameters,
                              set_active, set_backbone_trainable)
    from aste_data import load_auxiliary

    if arm not in FUSION_ARMS:
        raise ValueError(f"unknown arm {arm!r}; have {sorted(FUSION_ARMS)}")
    spec = FUSION_ARMS[arm]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    cfg = AsteConfig.preset(spec["preset"])
    backbone = AutoExtractor.from_pretrained(model_id)
    encoder = backbone.encoder
    tokenizer = backbone.processor.tokenizer
    cfg.hidden_size = encoder.config.hidden_size
    model = AsteModel(encoder, cfg).to(device)

    data, stats = {}, {}
    for split in ("train", "dev", "test"):
        data[split], stats[split] = encode(
            load_split(domain, split, data_root), tokenizer, max_length
        )
    aux_stats = {"kept": 0}
    if spec["use_aux"]:
        aux_examples, aux_stats = load_auxiliary(domain, data_root)
        data["aux"], stats["aux"] = encode(aux_examples, tokenizer, max_length)

    collate = AsteCollator(tokenizer.pad_token_id, cfg)
    loaders = {
        s: DataLoader(rows, batch_size=batch_size, shuffle=(s in ("train", "aux")),
                      collate_fn=collate)
        for s, rows in data.items()
    }

    enc_all = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_all]

    history, stage_report = [], {}
    if cfg.use_adapter:
        # Stage 1 is the shared adapter's job alone; the domain adapter is switched
        # off so it cannot absorb auxiliary-domain quirks that stage 2 would then
        # have to unlearn. Its up-projection is zero, so switching it off at that
        # point changes nothing numerically - it is about what receives gradient.
        set_backbone_trainable(model.encoder, False)
        set_active(model.encoder, "domain", False)
        shared = adapter_parameters(model.encoder, "shared")
        domain_p = adapter_parameters(model.encoder, "domain")
        stage1_groups = [{"params": shared, "lr": adapter_lr},
                         {"params": head, "lr": head_lr}]
    else:
        set_backbone_trainable(model.encoder, True)
        shared = domain_p = []
        stage1_groups = [{"params": backbone_parameters(model.encoder), "lr": encoder_lr},
                         {"params": head, "lr": head_lr}]

    # Each stage is weighted by the class balance of the corpus it trains on, so
    # the auxiliary stage sees the auxiliary pool's neutral share rather than the
    # target's. Recorded in the payload because on 15res the two differ by 13.7x
    # and that difference is the argument for running this arm at all.
    stage_polarity = {"target": polarity_counts(data["train"])}
    if spec["use_aux"]:
        stage_polarity["aux"] = polarity_counts(data["aux"])

    if spec["use_aux"] and stage1_epochs > 0:
        _only_trainable(model, stage1_groups)
        model.set_polarity_weights(stage_polarity["aux"])
        best1, hist1 = _fit(model, loaders["aux"], loaders["dev"], stage1_groups,
                            stage1_epochs, warmup_pct, device, "aux")
        history += hist1
        if best1 and best1["state"] is not None:
            model.load_state_dict(best1["state"])
            stage_report["stage1"] = {"best_epoch": best1["epoch"], "dev_f1": best1["f1"],
                                      "n_train": len(data["aux"])}

    if cfg.use_adapter:
        set_active(model.encoder, "domain", True)
        stage2_groups = [{"params": domain_p, "lr": adapter_lr},
                         {"params": head, "lr": head_lr}]
        if spec["train_shared_stage2"]:
            # A tenth of the adapter LR once stage 1 has run: stage 2 may correct
            # the shared adapter, it may not retrain it on 605-1265 sentences.
            lr = adapter_lr * 0.1 if spec["use_aux"] else adapter_lr
            stage2_groups.insert(0, {"params": shared, "lr": lr})
    else:
        stage2_groups = [{"params": backbone_parameters(model.encoder), "lr": encoder_lr},
                         {"params": head, "lr": head_lr}]

    _only_trainable(model, stage2_groups)
    # After stage 1's checkpoint is loaded, not before: ``polarity_weight`` is a
    # buffer and ``load_state_dict`` would otherwise put the auxiliary pool's
    # class balance back. The prototypes are buffers too, and those are meant to
    # carry over - that is the transfer this arm tests.
    model.set_polarity_weights(stage_polarity["target"])
    best2, hist2 = _fit(model, loaders["train"], loaders["dev"], stage2_groups,
                        stage2_epochs, warmup_pct, device, "target")
    history += hist2
    if best2 and best2["state"] is not None:
        model.load_state_dict(best2["state"])
    stage_report["stage2"] = {"best_epoch": best2["epoch"], "dev_f1": best2["f1"],
                              "n_train": len(data["train"])}

    test = evaluate(model, loaders["test"], device, keep_predictions=True)
    predictions = test.pop("predictions", [])
    grid = [round(0.05 * i, 2) for i in range(0, 19)]
    dev_curve = evaluate_sweep(model, loaders["dev"], device, grid)
    best_th = max(grid, key=lambda t: dev_curve[t]["f1"])
    test_curve = evaluate_sweep(model, loaders["test"], device, grid)

    return {
        "task": "aste_triplet_fusion", "domain": domain, "preset": arm, "arm": arm,
        "base_preset": spec["preset"], "seed": seed, "model_id": model_id,
        "config": asdict(cfg), "params": model.added_parameter_count(),
        "adapter_report": model.adapter_report,
        "epochs": stage2_epochs,
        "stage1_epochs": stage1_epochs if spec["use_aux"] else 0,
        "best_epoch": best2["epoch"], "dev_f1": best2["f1"],
        "stages": stage_report, "aux_stats": aux_stats,
        "stage_polarity": {k: {"POS": v[0], "NEU": v[1], "NEG": v[2]}
                           for k, v in stage_polarity.items()},
        "polarity_weight": ([round(float(v), 4) for v in model.polarity_weight]
                            if hasattr(model, "polarity_weight") else None),
        "encoder_lr": encoder_lr, "adapter_lr": adapter_lr, "head_lr": head_lr,
        "warmup_pct": warmup_pct,
        "test": test, "predictions": predictions, "history": history,
        "tuned_threshold": best_th, "test_tuned": test_curve[best_th],
        "dev_curve": {str(t): dev_curve[t] for t in grid},
        "test_curve": {str(t): test_curve[t] for t in grid},
        "data_stats": stats,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
