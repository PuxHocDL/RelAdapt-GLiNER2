"""Loader for SemEval ASTE-Data-V2 triplets.

Line format::

    sentence####[([aspect_word_idx], [opinion_word_idx], 'POS'), ...]

Word indices are positions in the whitespace-split sentence, so they are
converted here to character offsets once, up front. Every downstream consumer
(ACSI training, probes, evaluation) then works in character space, which is the
same space the Vietnamese span datasets already use.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DOMAINS = ("14lap", "14res", "15res", "16res")
SPLITS = ("train", "dev", "test")
POLARITIES = ("POS", "NEU", "NEG")
POLARITY_TEXT = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}


@dataclass
class Triplet:
    aspect: tuple[int, int]      # character span [start, end)
    opinion: tuple[int, int]     # character span [start, end)
    polarity: str                # POS | NEU | NEG

    def as_tuple(self) -> tuple:
        return (self.aspect, self.opinion, self.polarity)


@dataclass
class AsteExample:
    id: str
    text: str
    words: list[str]
    triplets: list[Triplet] = field(default_factory=list)

    @property
    def aspects(self) -> list[tuple[int, int]]:
        seen, out = set(), []
        for t in self.triplets:
            if t.aspect not in seen:
                seen.add(t.aspect)
                out.append(t.aspect)
        return out

    @property
    def opinions(self) -> list[tuple[int, int]]:
        seen, out = set(), []
        for t in self.triplets:
            if t.opinion not in seen:
                seen.add(t.opinion)
                out.append(t.opinion)
        return out


def _char_to_token_span(offsets, start: int, end: int) -> tuple[int, int] | None:
    """Smallest token range covering [start, end); None if it cannot be covered."""
    first = last = None
    for i, (s, e) in enumerate(offsets):
        if s == e:  # special token
            continue
        if e > start and s < end:
            if first is None:
                first = i
            last = i
    if first is None or last is None:
        return None
    return (first, last + 1)


def _word_char_spans(words: list[str]) -> list[tuple[int, int]]:
    """Character span of each word in ``" ".join(words)``."""
    spans, cursor = [], 0
    for word in words:
        spans.append((cursor, cursor + len(word)))
        cursor += len(word) + 1  # single joining space
    return spans


def _indices_to_span(indices: list[int], spans: list[tuple[int, int]]) -> tuple[int, int]:
    """Merge word indices into one character span.

    ASTE indices are contiguous per element; a non-contiguous list would silently
    become an over-wide span, so that is rejected rather than quietly accepted.
    """
    if not indices:
        raise ValueError("empty index list")
    lo, hi = min(indices), max(indices)
    if sorted(indices) != list(range(lo, hi + 1)):
        raise ValueError(f"non-contiguous indices {indices}")
    if hi >= len(spans):
        raise ValueError(f"index {hi} beyond {len(spans)} words")
    return (spans[lo][0], spans[hi][1])


def parse_line(line: str, example_id: str) -> AsteExample:
    sentence, _, raw = line.partition("####")
    if not raw:
        raise ValueError(f"{example_id}: missing '####' separator")
    words = sentence.split(" ")
    spans = _word_char_spans(words)
    text = " ".join(words)

    triplets = []
    for aspect_idx, opinion_idx, polarity in ast.literal_eval(raw.strip()):
        polarity = str(polarity).upper()
        if polarity not in POLARITIES:
            raise ValueError(f"{example_id}: unknown polarity {polarity!r}")
        triplets.append(
            Triplet(
                aspect=_indices_to_span(list(aspect_idx), spans),
                opinion=_indices_to_span(list(opinion_idx), spans),
                polarity=polarity,
            )
        )
    return AsteExample(id=example_id, text=text, words=words, triplets=triplets)


def load_split(domain: str, split: str, root: str | Path = "Data/ASTE") -> list[AsteExample]:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = Path(root) / domain / f"{split}_triplets.txt"
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            out.append(parse_line(line, f"{domain}_{split}_{i}"))
    return out


def load_domain(domain: str, root: str | Path = "Data/ASTE") -> dict[str, list[AsteExample]]:
    return {split: load_split(domain, split, root) for split in SPLITS}


def _norm_key(text: str) -> str:
    """Lowercase, alphanumeric-only, whitespace-collapsed form of a sentence."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _tok_key(text: str) -> frozenset:
    return frozenset(_norm_key(text).split())


def _is_near_duplicate(text: str, held_toks: list[frozenset], threshold: float) -> bool:
    a = _tok_key(text)
    if not a or threshold > 1.0:
        return False
    for b in held_toks:
        inter = len(a & b)
        if inter and inter / len(a | b) >= threshold:
            return True
    return False


def load_auxiliary(target: str, root: str | Path = "Data/ASTE",
                   splits: tuple[str, ...] = ("train", "dev"),
                   near_dup: float = 0.9) -> tuple[list[AsteExample], dict]:
    """Training sentences from the other three domains, with leakage removed.

    The four ASTE-Data-V2 domains are not disjoint corpora. Measured on this
    copy of the data:

        16res.train+dev  contains 100.0% of 15res.test  (322/322)
        16res.train+dev  contains 100.0% of 15res.dev   (148/148)
        15res.train+dev  contains  69.5% of 16res.dev
        14res.train+dev  contains  69.6% of 15res.dev

    So pooling the domains naively puts the whole of 15res's test set into its
    own training data. Every sentence that appears anywhere in the target's test
    or dev split is dropped here, as is anything already in the target's own
    train split, and the surviving pool is deduplicated by sentence text.

    Sizes after filtering (own train -> own train + clean auxiliary):

        14lap    900 -> 3,038      14res  1,265 -> 2,943
        15res    605 -> 2,785      16res    857 -> 3,043
    """
    if target not in DOMAINS:
        raise ValueError(f"unknown domain {target!r}; expected one of {DOMAINS}")
    held_ex = [e for split in ("test", "dev") for e in load_split(target, split, root)]
    held = {e.text for e in held_ex}
    own = {e.text for e in load_split(target, "train", root)}

    # Exact-string filtering is not enough on its own. Audited over all four
    # domains, it still lets through 1 sentence into 14res.test and 2 into
    # 16res.test that carry the identical bag of words and differ only in
    # punctuation or word order. Small, but it is real leakage, so the held-out
    # side is also indexed by a normalised token key, and anything at Jaccard
    # >= `near_dup` against a held-out sentence is dropped as well.
    held_keys = {_norm_key(e.text) for e in held_ex}
    held_toks = [_tok_key(e.text) for e in held_ex]
    held_toks = [t for t in held_toks if t]

    out, seen = [], set()
    stats = {"leaked": 0, "in_own_train": 0, "duplicate": 0, "near_duplicate": 0,
             "kept": 0, "per_domain": {}}
    for domain in DOMAINS:
        if domain == target:
            continue
        kept_here = 0
        for split in splits:
            for ex in load_split(domain, split, root):
                if ex.text in held or _norm_key(ex.text) in held_keys:
                    stats["leaked"] += 1
                elif ex.text in own:
                    stats["in_own_train"] += 1
                elif ex.text in seen:
                    stats["duplicate"] += 1
                elif _is_near_duplicate(ex.text, held_toks, near_dup):
                    stats["near_duplicate"] += 1
                else:
                    seen.add(ex.text)
                    out.append(ex)
                    kept_here += 1
        stats["per_domain"][domain] = kept_here
    stats["kept"] = len(out)
    stats["near_dup_threshold"] = near_dup
    return out, stats


def overlap_report(root: str | Path = "Data/ASTE") -> dict:
    """Cross-domain sentence overlap, the evidence behind ``load_auxiliary``."""
    sents = {(d, s): {e.text for e in load_split(d, s, root)}
             for d in DOMAINS for s in SPLITS}
    rows = []
    for x in DOMAINS:
        pool = sents[(x, "train")] | sents[(x, "dev")]
        for y in DOMAINS:
            if x == y:
                continue
            for s in ("test", "dev"):
                n = len(pool & sents[(y, s)])
                if n:
                    rows.append({"source": f"{x}.train+dev", "target": f"{y}.{s}",
                                 "overlap": n, "target_size": len(sents[(y, s)]),
                                 "pct": round(100 * n / len(sents[(y, s)]), 1)})
    aux = {}
    for y in DOMAINS:
        _, st = load_auxiliary(y, root)
        aux[y] = {"own_train": len(sents[(y, "train")]), "clean_aux": st["kept"],
                  "total": len(sents[(y, "train")]) + st["kept"], "leaked_dropped": st["leaked"]}
    return {"overlap": rows, "auxiliary": aux}


def verify(root: str | Path = "Data/ASTE") -> dict:
    """Round-trip every triplet: the character span must reproduce the words."""
    report: dict = {"domains": {}, "errors": []}
    totals = {"examples": 0, "triplets": 0, "multi_aspect": 0, "multi_word_aspect": 0}

    for domain in DOMAINS:
        per_domain = {}
        for split in SPLITS:
            examples = load_split(domain, split, root)
            n_trip = sum(len(e.triplets) for e in examples)
            multi = sum(1 for e in examples if len(e.aspects) > 1)
            for e in examples:
                for t in e.triplets:
                    a = e.text[t.aspect[0]:t.aspect[1]]
                    o = e.text[t.opinion[0]:t.opinion[1]]
                    if not a.strip() or not o.strip():
                        report["errors"].append(f"{e.id}: empty span a={a!r} o={o!r}")
                    if a != a.strip() or o != o.strip():
                        report["errors"].append(f"{e.id}: padded span a={a!r} o={o!r}")
                    if " " in a:
                        totals["multi_word_aspect"] += 1
            per_domain[split] = {
                "examples": len(examples),
                "triplets": n_trip,
                "multi_aspect_sentences": multi,
            }
            totals["examples"] += len(examples)
            totals["triplets"] += n_trip
            totals["multi_aspect"] += multi
        report["domains"][domain] = per_domain

    report["totals"] = totals
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
