"""H-series: the polarity decision, and the class that never gets predicted.

The measurement this module exists for
--------------------------------------
Every intervention in this project so far aimed at the span pathway or at
pairing. Decomposing the stored ``B2h`` predictions (5 seeds x 4 domains) by
*where* a gold triplet is lost says that is no longer where the headroom is:

| bucket                                     | 14lap | 14res | 15res | 16res |
|--------------------------------------------|------:|------:|------:|------:|
| pair correct, polarity wrong (% of misses)  | 29.6  | 23.1  | 26.5  | 31.6  |
| boundary off (% of misses)                  | 18.1  | 28.8  | 24.9  | 31.6  |

A polarity error is the only kind that costs twice - one false negative and one
false positive on the same triplet - so relabelling those, *without emitting a
single extra prediction*, is worth:

| oracle              | 14lap | 14res | 15res | 16res | avg   |
|---------------------|------:|------:|------:|------:|------:|
| B2h                 | 63.74 | 72.71 | 66.50 | 73.99 | 69.24 |
| + correct polarity  | 73.93 | 78.05 | 73.84 | 80.37 | 76.55 |
| + correct boundary  | 69.15 | 78.07 | 72.17 | 79.34 | 74.68 |

49-77% of the polarity bucket is one class. NEU is predicted **0 times** on
15res across five seeds against 125 gold, 5 times on 16res against 145, and its
per-class triplet F1 is 0.00 / 2.67 / 1.09 / 9.09. It is not that NEU is hard;
it is that NEU is never attempted.

Worse, the current leading arm caused it. B2h raised the weight on hard
negatives, which is weight on the `none` class, and NEU fell with it:

| domain | B2 NEU F1 | B2h NEU F1 | B2 -> B2h triplet F1 |
|--------|----------:|-----------:|---------------------:|
| 16res  |     30.49 |       2.67 |    72.46 -> 74.00    |
| 14res  |     24.69 |       9.09 |    71.30 -> 72.72    |
| 14lap  |      6.58 |       1.09 |    61.36 -> 63.74    |
| 15res  |      7.35 |       0.00 |    64.15 -> 66.49    |

B2h bought +1.90 F1 partly by giving up on the minority class. That trade is
real and worth keeping - but it is also a signpost: the class was cheap to
abandon because it had almost no gradient to begin with.

The three things here
---------------------
1. ``factored_pair_logits`` - split one 4-way softmax into P(exists) x
   P(polarity | exists). The *function class does not change*: softmax is
   shift-invariant, so pinning the none logit to 0 and letting `exist` absorb it
   is the same model, and ``pair_dim -> 1`` plus ``pair_dim -> 3`` has exactly
   the same parameter count as ``pair_dim -> 4``. What changes is the objective:
   NEU stops competing against `none` over 100 cells and starts competing
   against POS and NEG over the ~2 that are real.

2. ``SentimentPrototypes`` - a prototype contrastive term on the pair
   representation, the mechanism MiniConGTS (-2.12 to -7.04 F1 without it), MPGM
   (D2SCL) and GAC all share. Written against EMA class centroids rather than
   in-batch positives because at batch size 8 with NEU at 2-7% of triplets a
   batch usually holds zero or one NEU pair, and in-batch SupCon needs two.
   The centroids are buffers, so they cost no trainable parameters and they
   travel in the state dict - which is what carries the geometry learned on the
   auxiliary domains into the target domain in the fusion arm.

3. ``boundary_listwise_loss`` - the second bucket. Span scores are currently
   ~1000 independent per-span sigmoids, so `battery`, `battery life` and `life`
   are free to all score high. One softmax per start position over the possible
   end positions, plus a "no span starts here" slot, makes them compete. This is
   BDTF's relation region expressed as an objective instead of as a decoder, and
   like everything else here it adds no parameters.

Why this is the right place for the domain-fusion adapter
--------------------------------------------------------
NEU supervision is the scarcest thing in this corpus, and it is scarce *per
domain* rather than in total. Counted on the leakage-filtered auxiliary pool
from ``aste_data.load_auxiliary``:

| target | own NEU triplets | clean auxiliary NEU | ratio |
|--------|-----------------:|--------------------:|------:|
| 15res  |               25 |                 343 | 13.7x |
| 16res  |               50 |                 343 |  6.9x |
| 14lap  |              126 |                 258 |  2.0x |
| 14res  |              166 |                 199 |  1.2x |

15res cannot learn a neutral class from 25 examples. The shared adapter stage is
the only place in this pipeline where enough of them exist, and the sentiment
prototypes are what that stage hands over.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

POLARITY_NAMES = ("POS", "NEU", "NEG")


# ---------------------------------------------------------------------------
# 1. The factorisation
# ---------------------------------------------------------------------------
def factored_pair_logits(exist: torch.Tensor, polarity: torch.Tensor) -> torch.Tensor:
    """(B,Na,No) existence logit + (B,Na,No,3) polarity logits -> (B,Na,No,4).

    The 4-way tensor is rebuilt so every existing consumer - ``decode``,
    ``evaluate``, ``evaluate_sweep``, the dev threshold curve - keeps working
    unchanged, and so the comparison against B2h stays single-variable.

    ``none`` is pinned at 0 (softmax is shift-invariant, so ``exist`` absorbs it)
    and the polarity log-softmax is **max-subtracted** before it is added on.
    That last part is not cosmetic. The obvious composition,

        P(none) = 1/Z,   P(c) = exp(exist) * P(c | exists) / Z

    turns the decode rule ``max_c P(c) > P(none)`` into ``exist > -log P(c*)``,
    a penalty of up to 1.099 logits that grows as the polarity gets *less*
    certain. The existence head is trained with BCE, i.e. calibrated for
    ``exist > 0``, so that rule decodes a different decision from the one it
    optimises - and it bites hardest on the least confident class, which is NEU,
    which is the class this whole module exists to bring back. Subtracting the
    max leaves the polarity argmax untouched and makes the rule exactly
    ``sigmoid(exist) > 0.5``: existence decides existence, polarity decides the
    label, which is the factorisation the arm is testing.

    ``conf`` stays monotone in ``exist``, so the dev threshold sweep is still a
    single ordered knob; it is no longer a calibrated posterior, but neither the
    decode rule nor the sweep reads it as one.
    """
    log_pol = polarity.log_softmax(-1)
    log_pol = log_pol - log_pol.max(dim=-1, keepdim=True).values
    none = torch.zeros_like(exist).unsqueeze(-1)
    return torch.cat([none, exist.unsqueeze(-1) + log_pol], dim=-1)


def class_weights(counts, scheme: str, device=None):
    """Per-class weights for the 3-way head from training-split frequencies.

    ``counts`` is (n_POS, n_NEU, n_NEG). Applied inside the polarity head only:
    up-weighting NEU there cannot re-open the existence imbalance that
    ``none_weight`` exists to control, which is exactly what goes wrong when the
    same fix is attempted on the 4-way loss.
    """
    if scheme == "none":
        return None
    f = torch.tensor([max(int(c), 1) for c in counts], dtype=torch.float, device=device)
    if scheme == "balanced":
        w = f.sum() / f
    elif scheme == "sqrt":
        w = (f.sum() / f).sqrt()
    else:
        raise ValueError(f"unknown polarity_class_weight {scheme!r}")
    return w * (len(f) / w.sum())        # mean 1, so the loss scale is unchanged


# ---------------------------------------------------------------------------
# 2. Sentiment prototypes
# ---------------------------------------------------------------------------
class SentimentPrototypes(nn.Module):
    """EMA class centroids over pair representations, plus a ProtoNCE term.

    No trainable parameters: the centroids are buffers updated from detached
    representations, and gradient reaches the model only through the similarity
    of a live representation to a frozen centroid. That keeps the arm inside the
    rule this project has been following - added capacity has to come with a
    pretrained prior or with more data, and this adds no capacity at all.

    At step 0 every centroid is zero, ``F.normalize`` returns zero, every logit
    is zero and no class is ``ready``, so the term contributes exactly nothing
    and the arm starts bit-identical to the one it is built on.
    """

    def __init__(self, dim: int, n_classes: int = 3, momentum: float = 0.95,
                 temperature: float = 0.1):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("centroid", torch.zeros(n_classes, dim))
        self.register_buffer("seen", torch.zeros(n_classes))

    @property
    def ready(self) -> torch.Tensor:
        return self.seen > 0

    @torch.no_grad()
    def update(self, h: torch.Tensor, y: torch.Tensor) -> None:
        for c in y.unique():
            mean = h[y == c].mean(0)
            if self.seen[c] == 0:
                self.centroid[c] = mean
            else:
                m = self.momentum
                self.centroid[c] = m * self.centroid[c] + (1.0 - m) * mean
            self.seen[c] += 1

    def forward(self, h: torch.Tensor, y: torch.Tensor, weight=None) -> torch.Tensor:
        """h (M,D) pair representations of gold pairs, y (M,) polarity in 0..2."""
        if h.numel() == 0:
            return h.new_zeros(())
        z = F.normalize(h, dim=-1)
        p = F.normalize(self.centroid, dim=-1)
        logits = z @ p.t() / self.temperature
        # A class whose centroid was never filled is not a real negative: scoring
        # against a zero vector would push every anchor away from an arbitrary
        # direction. Mask it out of the denominator instead.
        logits = logits.masked_fill(~self.ready.view(1, -1),
                                    torch.finfo(logits.dtype).min)
        keep = self.ready[y]
        loss = h.new_zeros(())
        if bool(keep.any()):
            loss = F.cross_entropy(logits[keep], y[keep], weight=weight)
        if self.training:
            # Never let an evaluation pass move the centroids: the dev-selected
            # checkpoint has to be the same model that was scored.
            self.update(h.detach(), y)
        return loss


# ---------------------------------------------------------------------------
# 3. Losses
# ---------------------------------------------------------------------------
def existence_loss(exist: torch.Tensor, pair_target: torch.Tensor,
                   cell_weight: torch.Tensor) -> torch.Tensor:
    """Binary version of the pair loss, carrying B2h's structured negatives.

    ``cell_weight`` is built by the caller with exactly the rule B2h uses - 1.0
    on a positive, ``hard_none_weight`` on a negative whose aspect or opinion is
    gold, ``none_weight`` on the rest - so the existence half of the factorised
    model optimises the same objective on the same cells with the same weights.
    The only thing that leaves it is the polarity decision.
    """
    target = (pair_target > 0).to(exist.dtype)
    per_cell = F.binary_cross_entropy_with_logits(exist, target, reduction="none")
    return (per_cell * cell_weight).sum() / cell_weight.sum().clamp(min=1e-6)


def polarity_loss(polarity: torch.Tensor, pair_target: torch.Tensor,
                  weight=None) -> torch.Tensor:
    """3-way loss over the gold pairs only.

    This is the point of the factorisation. In the 4-way loss a NEU cell is one
    of ~2 positives among 100 cells and shares its gradient with the `none`
    class; here the denominator is the number of gold pairs in the batch, so NEU
    goes from roughly 0.04% of the pairing gradient to its share of the positive
    class - two orders of magnitude more.
    """
    pos = pair_target > 0
    if not bool(pos.any()):
        return polarity.new_zeros(())
    return F.cross_entropy(polarity[pos], pair_target[pos] - 1, weight=weight)


def boundary_listwise_loss(span_logits: torch.Tensor, gold: torch.Tensor,
                           spans: torch.Tensor, span_valid: torch.Tensor,
                           max_width: int, n_tokens: torch.Tensor) -> torch.Tensor:
    """One softmax per start position over end positions, plus a "no span" slot.

    Boundary errors are 18-32% of missed triplets and 27-32% of false positives,
    and the per-span sigmoid the model trains on cannot see them: `battery`,
    `battery life` and `life` are three independent questions with three
    independent answers. Grouping every span that shares a start into one softmax
    makes them one question with one answer. No parameters.

    span_logits (B,N), gold (B,N) bool, spans (N,2), span_valid (B,N),
    n_tokens (B,) unpadded lengths.
    """
    b, n = span_logits.shape
    device = span_logits.device
    starts = spans[:, 0]
    widths = spans[:, 1] - spans[:, 0] - 1                     # 0-based slot
    t = int(starts.max().item()) + 1
    neg = torch.finfo(span_logits.dtype).min

    flat = (starts * (max_width + 1) + widths).view(1, n).expand(b, n)
    grid = span_logits.new_full((b, t * (max_width + 1)), neg)
    grid = grid.scatter(1, flat, span_logits.masked_fill(~span_valid, neg))
    grid = grid.view(b, t, max_width + 1).clone()
    grid[:, :, max_width] = 0.0                                # "no span here"

    target = torch.full((b, t), max_width, dtype=torch.long, device=device)
    idx = gold.nonzero(as_tuple=False)
    if idx.numel():
        target[idx[:, 0], starts[idx[:, 1]]] = widths[idx[:, 1]]

    valid_start = torch.arange(t, device=device).view(1, t) < n_tokens.view(b, 1)
    lp = grid.log_softmax(-1)
    nll = -lp.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return (nll * valid_start).sum() / valid_start.sum().clamp(min=1)
