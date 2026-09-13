# Pre-registration: canonical B2h vs R1bm under the RNG-fixed build

Written before any run on seeds 62-66 existed.

## Why this has to be re-run

R1bm's +1.22 replication (seeds 52-56) was measured under a build where
constructing an extra submodule consumed global RNG and therefore shifted the
`DataLoader` shuffle permutation. R1bm has a biaffine module that B2h does not,
so **the two arms trained on different data orders** - not just different
initialisations. Every earlier B2h-vs-R1bm comparison in this project carries
that confound.

How much it can matter is already measured: the same R1bm config scored 72.89 on
seeds 57-59 before the fix and 73.62 after, on identical seeds. That is ~0.7 F1
of pure data-order movement, against a claimed effect of +1.22.

So the headline result is re-measured from scratch under the clean build. If it
survives, R1bm is the new baseline on solid ground. If it collapses toward zero,
most of the earlier gain was confound and must be retracted.

## The invariant that makes this checkable

Each run now records `batch_order_hash`: a SHA-256 over the example ids of the
first 10 minibatches of epoch 0, taken from the real training loop (not a probe
iterator, which would itself draw the permutation and change training).

**Requirement: for every seed, `batch_order_hash` must be identical for B2h and
R1bm.** Verified to be sensitive - a shifted RNG produces a different hash. If
any seed disagrees, that seed is void and the confound has returned.

## Design, frozen

| | |
|---|---|
| arms | exactly two: `B2h`, `R1bm` |
| domains | 14lap, 14res, 15res, 16res |
| seeds | 62, 63, 64, 65, 66 - disjoint from 42-46, 47-51, 52-56, 57-59 |
| split | **dev only**. Test is not read. |
| epochs / batch | 20 / 8 |
| encoder LR / head LR / warmup | 2e-5 / 1e-4 / 0.2, OneCycle cosine |
| pruning | `n_aspect_keep 10`, `n_opinion_keep 10`, `max_span_width 8` |
| threshold | untuned decode (0.0) |
| loss | unchanged from B2h in both arms |
| build | RNG state restored at the end of `AsteModel.__init__` |
| total | 2 x 4 x 5 = 40 runs |

Nothing is tuned on these seeds for either arm.

## Decision rule, fixed in advance

**R1bm survives** if both hold:

1. pooled `dF1 >= +0.40` averaged over the four domains, and
2. at least **3 of 4** domains positive.

Interpretation bands agreed in advance:

- `+0.8` to `+1.2` with 4/4 positive -> R1bm is the new baseline on solid ground
- near zero -> most of the earlier gain was trajectory/data-order confound, and
  the +1.22 and the +0.93 test number are **retracted**

## If R1bm survives

No new architecture branch. The next step is a **cross-seed prediction stability
analysis**, which costs no training: over the 5 seeds, give every predicted
triplet a consensus count `c(y)` in 1..5, then report precision / recall / F1
when keeping only triplets with `c(y) >= k` for `k = 1..5`.

The hypothesis this tests is that true positives are stable across seeds while
false positives are not - which would be consistent with everything measured so
far: R1bm wins by selectivity, S1 can identify false positives but only at a
recall cost, and data order alone moves F1 by ~0.7. If the consensus curve
separates cleanly, the follow-up is consistency-based selectivity (ultimately
distilling consensus into a single-model confidence head), not more context or
more loss terms.
