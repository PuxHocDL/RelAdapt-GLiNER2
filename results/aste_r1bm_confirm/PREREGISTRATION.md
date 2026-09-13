# Pre-registration: R1bm vs B2h, fresh-seed dev replication

Written before any run on seeds 52-56 existed. The exploratory evidence is on
seeds 42-46 and is not reused here except to state the hypothesis.

## Hypothesis

> Replacing the concatenation MLP in the relation scorer with a biaffine
> interaction - `s(a,o) = h_a^T W h_o + U[h_a; h_o] + b` added to the existing
> head, plus relative distance, span widths and `|h_a - h_o|` - improves exact
> triplet F1 over B2h.

Directional: R1bm > B2h.

## Why a replication is needed

R1bm was chosen from a 5-arm dev screen (R1m, R1b, R1bm, R1bf against B2h) on
seeds 42-46, and its dev margin over the runner-up R1b was **0.007 F1**
(71.3330 vs 71.3263) - a tie broken arbitrarily by iteration order. A single
exploratory screen that ends in a coin flip is not evidence; this run is.

## Frozen, exactly as run in the exploratory screen

| | |
|---|---|
| arms | exactly two: `B2h`, `R1bm`. Nothing else runs on these seeds. |
| preset definition | `AsteConfig.preset("R1bm")`, unmodified |
| scorer | biaffine `nn.Bilinear(256, 256, 4)` + `nn.Linear(512, 4)`, bilinear weight zero-initialised |
| features | `use_pair_diff`, `use_pair_width`, `use_distance_feature` all on |
| pair_dim / span_dim | 256 / 256 |
| loss | B2h's: `w_aspect 1.0 + w_opinion 1.0 + w_pair 1.0`, `hard_none_weight 1.0`, `none_weight 0.1` |
| assignment | Sinkhorn, 20 iters, `assignment_weight 1.0`, temp 1.0 -> 1.0 |
| pruning | `n_aspect_keep 10`, `n_opinion_keep 10`, `max_span_width 8` |
| threshold | untuned decode (0.0); the tuned number is secondary |
| epochs / batch | 20 / 8 |
| encoder LR / head LR / warmup | 2e-5 / 1e-4 / 0.2, OneCycle cosine |
| model | `fastino/gliner2-base-v1` |
| seeds | 52, 53, 54, 55, 56 - disjoint from 42-46 and from 47-51 |
| split scored | **dev only**. Test is not read for this experiment. |
| GPU | A10G |

No hyperparameter is tuned on these seeds for either arm. No biaffine variant,
no additional feature combination, no hidden-size search.

## Decision rule, fixed in advance

**R1bm survives** if both hold on dev:

1. pooled `dF1 >= +0.40` averaged over the four domains, and
2. at least **3 of 4** domains positive.

Otherwise it does not survive, and the biaffine line is closed rather than
re-tuned.

## Secondary, reported but not decisive

Precision, recall, deep-rejection rate (share of gold relation cells at
existence log-odds < -2), emit recall given both spans survive pruning, and the
per-domain direction.

## What this cannot establish

The test split has already been read many times in this project, so even a
successful replication here leaves the final test number indicative rather than
a clean held-out result. This experiment establishes that the dev effect is
real and not a selection artefact; it does not launder the test split.
