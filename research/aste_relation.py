"""L-series: relation existence as a candidate-set decision, not 100 independent ones.

Why this and not another span objective
---------------------------------------
The decoder funnel on dev (B2h, 4 domains x 5 seeds, 7,555 gold triplets, buckets
disjoint and summing to n_gold) locates the failure precisely:

| stage                                        | % of gold triplets |
|----------------------------------------------|-------------------:|
| span never enumerable                        | 0.20 |
| pruned, an overlapping rival survived instead| 3.68 |
| pruned, no rival                             | 1.05 |
| **survived pruning, pair rejected as NONE**  | **16.94** |
| survived, wrong polarity                     | 6.70 |
| hit                                          | 71.44 |

95.3% of gold triplets have both spans alive after pruning, raw overlap ranking
is already 92-95%, and pair *identity* is 97.6-99.1% once a relation is emitted
at all. What fails is **relation existence**: the classifier scores each of ~100
cells as an independent yes/no and sends a gold pair to NONE.

The three are different questions and had been conflated:

    pair identity  !=  pair existence  !=  polarity

The multi-positive requirement
------------------------------
A single softmax over [o_1 ... o_n, NONE] per aspect is the obvious listwise
form and it is **wrong here**. Measured on train:

    P(deg(aspect) > 1) = 12.9%     P(deg(opinion) > 1) = 8.2%     max degree 4

so for one aspect in eight, two gold opinions would be forced to share the mass
of one distribution and compete against each other. ``aste_full``'s existing
``use_listwise`` (presets B2hl/B2hl1) does exactly that - it sums -log p over
every positive inside one log_softmax - and is therefore mis-specified for 12.9%
of aspects.

This module implements the multi-positive form instead: every gold partner gets
its own normalisation against the negatives and NONE, and never against another
gold partner.

    L_a = -(1/|P_a|) sum_{o+ in P_a}
              log  e^{s(a,o+)} / ( e^{s(a,o+)} + e^{s(a,NONE)} + sum_{o- in N_a} e^{s(a,o-)} )

With ``s(a,NONE)`` pinned at 0 (the existence logit is already a log-odds against
NONE) that collapses to a softplus, which is what the code computes:

    loss(o+) = softplus( logaddexp(0, logsumexp_{o-} s(a,o-)) - s(a,o+) )

and a row with no gold partner at all is trained toward NONE:

    loss(empty row) = softplus( logsumexp_{o-} s(a,o-) )

No parameters. It changes the unit of the decision from a cell to a candidate
set, which is the intervention the funnel argues for.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NEG = -1e9          # finite, so logsumexp over an all-masked row stays finite


def _row_loss(e: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """One side of the decision. e (B,R,C) existence log-odds, pos/valid (B,R,C)."""
    neg_cells = valid & ~pos
    lse_neg = torch.logsumexp(e.masked_fill(~neg_cells, NEG), dim=-1)      # (B,R)
    # Denominator shared by every positive in the row: NONE (logit 0) plus every
    # negative. Gold partners are deliberately absent from it.
    denom = torch.logaddexp(torch.zeros_like(lse_neg), lse_neg)            # (B,R)

    per_pos = F.softplus(denom.unsqueeze(-1) - e)                          # (B,R,C)
    n_pos = pos.sum(-1)                                                    # (B,R)
    pos_loss = (per_pos * pos).sum(-1) / n_pos.clamp(min=1)

    # A row whose candidate set holds no gold partner should choose NONE.
    empty_loss = F.softplus(lse_neg)

    has_pos = n_pos > 0
    row_has_candidates = valid.any(-1)
    loss = torch.where(has_pos, pos_loss, empty_loss)
    return (loss * row_has_candidates).sum() / row_has_candidates.sum().clamp(min=1)


def multi_positive_listwise(pair_logits: torch.Tensor, pair_target: torch.Tensor,
                            symmetric: bool = True) -> torch.Tensor:
    """Relation-existence loss over candidate sets, rows and columns.

    ``pair_logits`` is the (B,Na,No,4) tensor; existence is marginalised over the
    three polarities, so this objective says nothing about which sentiment a
    relation carries - that stays with the polarity part of the head.
    """
    e = torch.logsumexp(pair_logits[..., 1:], dim=-1) - pair_logits[..., 0]
    pos = pair_target > 0
    valid = torch.ones_like(pos)
    row = _row_loss(e, pos, valid)                       # aspect -> opinions
    if not symmetric:
        return row
    col = _row_loss(e.transpose(1, 2), pos.transpose(1, 2), valid.transpose(1, 2))
    return 0.5 * (row + col)


@torch.no_grad()
def existence_diagnostics(pair_logits: torch.Tensor, pair_target: torch.Tensor) -> dict:
    """What the existence decision looks like on gold cells that survived pruning.

    Returns counters, to be summed across batches:

    ``emitted``          gold pairs the model would emit (existence log-odds > 0)
    ``beats_none``       same thing, named for the comparison it is
    ``beats_rivals``     gold pair outscores every non-gold cell in its row
    ``both``             beats its rivals *and* NONE
    ``m_*``              margin histogram over gold cells, s(a,o) - s(NONE)

    The split matters: if a gold pair already beats every wrong opinion but loses
    to NONE, the model knows which opinion belongs and the *existence* decision
    is what is miscalibrated - which is exactly the case a candidate-set
    formulation fixes, and a case no amount of pair-identity modelling would.
    """
    e = torch.logsumexp(pair_logits[..., 1:], dim=-1) - pair_logits[..., 0]
    pos = pair_target > 0
    out = {"n_gold_cells": int(pos.sum())}
    if out["n_gold_cells"] == 0:
        return out
    m = e[pos]
    out["emitted"] = int((m > 0).sum())
    out["beats_none"] = out["emitted"]
    for name, lo, hi in (("m_gt0", 0.0, float("inf")), ("m_near", -0.5, 0.0),
                         ("m_mid", -2.0, -0.5), ("m_far", float("-inf"), -2.0)):
        out[name] = int(((m > lo) & (m <= hi)).sum()) if hi != float("inf") \
            else int((m > lo).sum())
    # Best non-gold cell in the same row (same aspect, a different opinion).
    rival = e.masked_fill(pos, NEG).max(dim=2, keepdim=True).values.expand_as(e)
    out["beats_rivals"] = int(((e > rival) & pos).sum())
    out["both"] = int(((e > rival) & (e > 0) & pos).sum())
    return out


# ---------------------------------------------------------------------------
# gold > NULL > negative
# ---------------------------------------------------------------------------
def null_rank_loss(pair_logits: torch.Tensor, pair_target: torch.Tensor,
                   m_pos: float = 1.0, m_neg: float = 1.0, alpha: float = 1.0,
                   hard: bool = False) -> torch.Tensor:
    """Put NULL in the middle instead of pushing it out of the way.

    ``B2hl`` failed because a single softmax over [opinions, NONE] has its
    optimum at p(o+) = 1/|P_a| and p(NONE) = 0: it learns "positives up, NONE
    away", which raises gold-pair emission (the rejected bucket fell in all four
    domains) and destroys precision (-1.92 dev F1, -4.64 at twice the weight).

    The decision the decoder actually makes is a threshold on the existence
    log-odds, so NULL is the boundary, not a competitor. This asks for a margin
    on *both* sides of it:

        e(a,o+) >  +m_pos        a gold relation sits above NULL
        e(a,o-) <  -m_neg        a wrong relation sits below NULL

    with ``e = logsumexp(polarity logits) - none logit``, so NULL is pinned at 0
    and neither side can win by moving it. Parameter-free.

    ``hard`` restricts each row and column to its hardest positive and hardest
    negative, which is much closer to what the decoder has to get right than
    summing over ~100 mostly-easy cells.
    """
    e = torch.logsumexp(pair_logits[..., 1:], dim=-1) - pair_logits[..., 0]
    pos = pair_target > 0
    neg = ~pos
    if not hard:
        n_p = pos.sum().clamp(min=1)
        n_n = neg.sum().clamp(min=1)
        l_pos = (F.softplus(m_pos - e) * pos).sum() / n_p
        l_neg = (F.softplus(m_neg + e) * neg).sum() / n_n
        return l_pos + alpha * l_neg

    def side(ee, pp, nn):
        # hardest positive = lowest-scoring gold cell; hardest negative = highest
        worst_pos = ee.masked_fill(~pp, 1e9).min(dim=-1).values
        worst_neg = ee.masked_fill(~nn, -1e9).max(dim=-1).values
        has_p, has_n = pp.any(-1), nn.any(-1)
        lp = (F.softplus(m_pos - worst_pos) * has_p).sum() / has_p.sum().clamp(min=1)
        ln = (F.softplus(m_neg + worst_neg) * has_n).sum() / has_n.sum().clamp(min=1)
        return lp + alpha * ln

    row = side(e, pos, neg)
    col = side(e.transpose(1, 2), pos.transpose(1, 2), neg.transpose(1, 2))
    return 0.5 * (row + col)


def degree_loss(pair_logits: torch.Tensor, pair_target: torch.Tensor) -> torch.Tensor:
    """Keep the expected number of emitted relations per row at the gold degree.

    Baseline under-emits (16.94% of gold relations rejected); B2hl over-emitted.
    This regularises the total relation mass rather than any individual decision,
    which is the quantity both failures get wrong in opposite directions.
    """
    e = torch.logsumexp(pair_logits[..., 1:], dim=-1) - pair_logits[..., 0]
    p = torch.sigmoid(e)
    gold = (pair_target > 0).to(p.dtype)
    row = (p.sum(-1) - gold.sum(-1)).abs().mean()
    col = (p.sum(1) - gold.sum(1)).abs().mean()
    return 0.5 * (row + col)
