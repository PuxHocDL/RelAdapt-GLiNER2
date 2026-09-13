# Pre-registration: consensus >= 3/5 replication on fresh seeds

Written before any run on seeds 67-71 existed.

## What is being replicated

On R1bm seeds 62-66 (dev, RNG-fixed build), taking every triplet predicted by at
least 3 of the 5 seeds gave:

| | P | R | F1 |
|---|---:|---:|---:|
| single seed, mean | - | - | **71.46** |
| consensus >= 3/5 | 74.66 | 73.68 | **74.11** |

i.e. **+2.65 F1**, precision +9.0 for recall -0.75. And the stability signal
behind it: `E[c | TP] = 4.44` against `E[c | FP] = 2.44`; 40.4% of false
positives are predicted by exactly one seed while 75.4% of true positives are
predicted by all five.

**`k = 3` was discovered on seeds 62-66 and is now FROZEN.** It is not re-tuned
here, and no other `k` is reported as the headline. That is the whole point of
this run: the k-curve was exploratory, this is not.

## Design, frozen

| | |
|---|---|
| arm | `R1bm` only, unchanged |
| seeds | 67, 68, 69, 70, 71 - disjoint from every seed used so far (42-46, 47-51, 52-56, 57-59, 62-66) |
| domains | 14lap, 14res, 15res, 16res |
| split | **dev first.** Test is read only if the dev replication passes. |
| consensus rule | `k = 3`, frozen |
| everything else | identical to the R1bm canonical run |
| runs | 1 x 4 x 5 = 20 |

## Decision rule, fixed in advance

The consensus effect **replicates** if, on fresh dev:

    F1(consensus >= 3/5) - mean F1(single seed)  >=  +2.00

Below +2.00 but above +1.00: partial replication, report as a weaker effect and
do not build on it without a third confirmation. Below +1.00: the consensus
effect does not replicate and the distillation plan is dropped.

## What gets read afterwards, and what it may be called

If dev replicates, the ensemble's **test** number is read once. It must be
reported as a **5-model ensemble at ~5x inference cost**, never as single-model
GLiNER2. The single-model comparison point stays the mean single-seed F1.

Expected test range, stated in advance so it cannot be rationalised afterwards:
the dev-test gap in this project is **not** a uniform optimism bias - per domain
it runs +0.45, +3.79, -5.62, -4.28, i.e. split difficulty, not overfitting. What
transfers is the *delta between arms*, and the one measurement of that is R1bm's
+1.53 dev -> +0.93 test, about 61%. So +2.65 dev at 60-100% transfer on a ~70.3
single-model test baseline predicts roughly **71.9-73.0**.

## After this, regardless of outcome

No further ensemble optimisation. The next step is **consensus distillation** -
train a stability head on `q(y) = c(y)/5` so a single model reproduces the
ensemble's selectivity at 1x inference. Recovering 60-70% of +2.65 would put a
single model at about **73.0-73.4**.

Explicitly not pursued: GCN, marker re-encoding, further loss reweighting,
pair-query context. All four have direct negative evidence in this project.

## Budget note

Four of the five Modal workspaces are out of billing budget. This run uses
`gliner-absa5`, the last one alive. If it exhausts mid-sweep the run is
incomplete and will be reported as such rather than averaged over whatever
finished.
