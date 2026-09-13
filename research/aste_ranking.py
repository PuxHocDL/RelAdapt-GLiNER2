"""K-series: overlap-aware exact-span ranking.

The question this asks is different from H0b's, and the difference is the point.

H0b asked *does the gold boundary look correct* - a listwise softmax over end
positions per start position, which is still a question about absolute
plausibility. Measured afterwards, it changed the boundary error rate by nothing
(partial-match rate 3.7 -> 3.7, 3.9 -> 3.7, 4.4 -> 3.8, 3.6 -> 4.0) and its
average gain did not replicate on fresh seeds.

This asks instead: **among the locally plausible overlapping boundaries, is the
exact span ranked first?** That is the failure the snap-boundary oracle actually
exposes - +6.90 F1 sits in predictions that already land on an overlapping span
and pick the wrong member of the local set.

Measured shape of that failure, on B2h over four domains and five seeds
(n = 1,689 partially matched gold spans):

| relation of the winning rival to the gold span | share |
|---|---:|
| left-truncated  | 33.7% |
| right-truncated | 25.8% |
| left-expanded   | 24.1% |
| right-expanded  | 13.3% |
| both edges wrong | 3.2% |

**96.9% of boundary errors move exactly one edge**, and the direction is
systematic: multi-token gold spans are truncated (85%), single-token gold spans
are over-extended (100%). The model's span extent regresses toward the middle.
So the competition that matters is with single-edge neighbours, which is why
``single_edge_only`` exists as an arm rather than a micro-optimisation - it is a
hypothesis about which rivals carry the signal, and it makes the rival set about
an order of magnitude smaller.

The objective
-------------
For a gold span g with overlapping rivals N(g),

    L_rank = sum_g log( 1 + sum_{c in N(g)} exp( s(c) - s(g) + m ) )

which is 0 when every rival sits at least ``m`` below the gold span and grows
smoothly otherwise. No parameters: it reweights the existing span logits.

Other gold spans of the same role are excluded from N(g) - two gold spans must
not be asked to rank above each other.

``overlap_rank_accuracy`` is the mechanism metric, and it is the one to read
before the F1: the share of gold spans that outscore every overlapping rival.
If that does not move, the arm did not do what it claims, whatever F1 says.
"""

from __future__ import annotations

import torch


def rival_mask(spans: torch.Tensor, single_edge_only: bool = False) -> torch.Tensor:
    """(N,N) bool. ``m[j,k]`` is True when span k is an overlapping rival of j.

    Spans are half-open [start, end), so j and k overlap iff
    ``s_j < e_k and s_k < e_j``.
    """
    s, e = spans[:, 0], spans[:, 1]
    ov = (s.unsqueeze(1) < e.unsqueeze(0)) & (s.unsqueeze(0) < e.unsqueeze(1))
    ov = ov & ~torch.eye(spans.shape[0], dtype=torch.bool, device=spans.device)
    if single_edge_only:
        shares_edge = (s.unsqueeze(1) == s.unsqueeze(0)) | (e.unsqueeze(1) == e.unsqueeze(0))
        ov = ov & shares_edge
    return ov


def _gather(scores, gold, spans, span_valid, single_edge_only):
    idx = gold.nonzero(as_tuple=False)                 # (G,2) -> [batch, span]
    if idx.numel() == 0:
        return None
    b_idx, j_idx = idx[:, 0], idx[:, 1]
    rivals = rival_mask(spans, single_edge_only)[j_idx]         # (G,N)
    # A rival must be a real span in this example, and must not itself be gold:
    # two gold spans of the same role are not competitors.
    rivals = rivals & span_valid[b_idx] & ~gold[b_idx]
    return scores[b_idx], scores[b_idx, j_idx], rivals


def overlap_ranking_loss(scores: torch.Tensor, gold: torch.Tensor,
                         spans: torch.Tensor, span_valid: torch.Tensor,
                         margin: float = 1.0,
                         single_edge_only: bool = False) -> torch.Tensor:
    """scores (B,N) span logits, gold (B,N) bool, spans (N,2), span_valid (B,N)."""
    got = _gather(scores, gold, spans, span_valid, single_edge_only)
    if got is None:
        return scores.new_zeros(())
    s_all, s_gold, rivals = got
    neg = torch.finfo(scores.dtype).min
    terms = (s_all - s_gold.unsqueeze(1) + margin).masked_fill(~rivals, neg)
    # log(1 + sum exp(terms)); a gold span with no rivals contributes exactly 0.
    lse = torch.logsumexp(terms, dim=1)
    return torch.logaddexp(torch.zeros_like(lse), lse).mean()


@torch.no_grad()
def overlap_rank_accuracy(scores: torch.Tensor, gold: torch.Tensor,
                          spans: torch.Tensor, span_valid: torch.Tensor,
                          single_edge_only: bool = False) -> tuple[int, int]:
    """(wins, total): gold spans that outscore every overlapping rival.

    The mechanism metric. Unlike triplet F1 it is insensitive to pairing, to
    polarity and to how many candidates survive pruning, so it isolates exactly
    the competition this objective claims to fix.
    """
    got = _gather(scores, gold, spans, span_valid, single_edge_only)
    if got is None:
        return 0, 0
    s_all, s_gold, rivals = got
    neg = torch.finfo(scores.dtype).min
    best = s_all.masked_fill(~rivals, neg).max(dim=1).values
    # A gold span with no rival wins by default: there is nothing to outrank.
    has_rival = rivals.any(dim=1)
    wins = ((s_gold > best) | ~has_rival).sum().item()
    return int(wins), int(s_gold.shape[0])
