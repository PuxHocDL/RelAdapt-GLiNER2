# Polarity-aware training on ASTE: a negative result

**Closed 2026-09-12.** `gliner2-base-v1`, 4 domains x 5 seeds (42-46) per arm,
control `B2h`. No further tuning of this line: no class-weight search, no NEU
decision threshold, no focal loss, no re-run at another `w_polarity`.

## The claim that was tested

Decomposing B2h's stored predictions said a gold pair that is found and paired
correctly but labelled with the wrong polarity is 23.1-31.6% of all missed
triplets and 17.0-26.7% of all false positives, and that relabelling those -
emitting nothing extra - would be worth +5.34 to +10.19 triplet F1. 49-77% of
the bucket was one class: B2h predicts NEU **0 times** on 15res against 125
gold, 5 times on 16res against 145.

So: factorise the 4-way pair softmax into `P(exists) x P(polarity | exists)`
(`H1`), optionally weight the three classes by training frequency and add an EMA
prototype contrastive term on the pair representation (`H3`). Neither adds a
trainable parameter - `pair_dim -> 1` plus `pair_dim -> 3` is exactly
`pair_dim -> 4`, verified at 800,396 head parameters for B2h, H1 and H3 alike.

## What happened

The intervention does what it was designed to do at the level of behaviour.

| 15res | NEU predictions | NEU F1 | POS F1 | NEG F1 | triplet F1 |
|---|---:|---:|---:|---:|---:|
| B2h | 0 / 125 | 0.00 | 72.73 | 57.86 | 66.49 |
| H1 | 87 / 125 | **33.96** | 70.75 | 55.69 | 65.09 |
| H3 | 55 / 125 | **37.78** | 70.60 | 54.89 | 65.24 |

It does not convert into end-task F1.

| arm | 14res | 14lap | 16res | 15res | avg |
|---|---:|---:|---:|---:|---:|
| B2h | 72.72 | 63.74 | 74.00 | 66.49 | **69.24** |
| H1 | 72.63 | 62.45 | 73.76 | 65.09 | 68.48 |
| H3 | 73.13 | 63.21 | 73.05 | 65.24 | 68.66 |

Paired bootstrap against B2h, 10,000 resamples of test sentences, Holm across
the four domains: **no domain reaches significance for either arm**, and the
diffs are negative in 6 of 8 domain-arm cells.

| | 14res | 14lap | 16res | 15res |
|---|---|---|---|---|
| H1 | -0.0009 (p 0.89) | -0.0129 (p 0.07) | -0.0024 (p 0.77) | -0.0139 (p 0.14) |
| H3 | +0.0042 (p 0.52) | -0.0053 (p 0.51) | -0.0095 (p 0.33) | -0.0124 (p 0.21) |

## Why, arithmetically

A NEU prediction that is right is +1 true positive. A NEU prediction that is
wrong almost always overwrites a POS/NEG triplet the control already had right,
so it is -1. **The arm nets positive only above 50% NEU precision.**

| domain | own NEU train examples | H1 NEU precision | H1 NEU recall | dF1 |
|---|---:|---:|---:|---:|
| 14res | 166 | 0.20 | 0.18 | -0.09 |
| 14lap | 126 | 0.12 | 0.05 | -1.29 |
| 16res | 50 | **0.51** | 0.28 | -0.24 |
| 15res | 25 | 0.41 | 0.29 | -1.39 |

Every domain sits at or below the break-even point.

## A prediction this experiment refuted

Before running the other three domains, the stated hypothesis was that NEU
precision is data-limited and would therefore rise with the number of neutral
training examples, making 14res (166) the best case and 15res (25) the worst.

**The correlation is negative.** The two domains with the most neutral data have
the worst neutral precision. Whatever caps NEU precision here, it is not simply
the count of neutral examples in the training split, which also removes the main
argument for routing this through the auxiliary-data adapter.

## The fusion arms, run and also negative

15res, 5 seeds, two-stage schedule (10 epochs auxiliary, 20 target):

| arm | mean F1 | runs that collapsed to ~0 |
|---|---:|---|
| F0 adapters, no auxiliary stage | 14.69 | 2 / 5 |
| F1 adapters + auxiliary stage | 27.37 | 3 / 5 |
| **F2 no adapters, same two stages, plain fine-tuning** | **66.61** | 0 / 5 |
| FH0 H3 + adapters | 0.33 | 5 / 5 |
| FH1 H3 + adapters + auxiliary stage | 16.20 | 3 / 5 |

Two separate findings, both negative:

1. **The D1 adapter design fails.** A frozen backbone with stacked shared/domain
   bottleneck adapters collapses to near-zero predictions on most seeds. This is
   the failure mode `none_weight` was originally introduced to prevent, and
   freezing the encoder reintroduces it.
2. **The auxiliary data on its own buys nothing.** F2, which is the clean
   control - same corpus, same schedule, no adapters, no collapses - reaches
   66.61 against B2h's 66.49 on the same domain. That is +0.12, inside seed
   noise, while training on 2,785 sentences instead of 605.

## The conclusion, stated no more strongly than the evidence supports

> Polarity-aware training substantially increases neutral-class recall and
> prediction coverage, but does not improve ASTE F1, because neutral precision
> remains below the break-even point at which a neutral prediction pays for the
> majority-class triplet it displaces.

What this does **not** establish: that neutral polarity is unlearnable on
ASTE-Data-V2, or that no representation or objective could separate it. Four
specific interventions (factorised decision, two class-weighting schemes, a
prototype contrastive term, and an auxiliary-data pre-stage) did not solve
neutral discrimination on these four domains. A different representation, a
different supervision signal, or better-specified neutral annotation could still
do so; this experiment does not speak to that.

## What it does establish, usefully

Polarity accuracy **conditional on both spans being exactly right** is already
86.2 / 93.2 / 90.1 / 92.1 on 14lap / 14res / 15res / 16res, and the pairing rate
conditional on the same is 97.8-98.9%. `P(sentiment | a, o)` is not where the
remaining F1 is. The snap-boundary oracle puts 6.34-7.71 F1 in span localisation
instead. That is the argument for moving the budget upstream.

## Reproduction

```
GLINER_ABSA_DOMAINS=14res,14lap,16res,15res GLINER_ABSA_PRESETS=H1,H3,H0b \
GLINER_ABSA_SEEDS=42,43,44,45,46 python scripts/launch_h_sweep.py
GLINER_ABSA_DOMAINS=15res GLINER_ABSA_ARMS=F0,F1,F2,FH0,FH1 \
python scripts/launch_fusion_sweep.py
python research/aste_stats.py results/aste_15res B2h
python scripts/aste_h_report.py
```

Presets `H1 H2 H2b H3 H4 FH FH2` stay defined in `aste_full.py` so these runs
remain reproducible. They are inert configuration and should not be extended.
