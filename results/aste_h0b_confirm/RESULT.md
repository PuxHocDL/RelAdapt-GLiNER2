# H0b: an exploratory finding that did not replicate

**Verdict: NOT CONFIRMED**, by the decision rule written in
`PREREGISTRATION.md` before these runs existed.

Two independent lines of evidence agree, which is what makes this conclusive
rather than merely disappointing: the effect did not reappear on fresh seeds,
**and** the mechanism it was supposed to work by was never detectable even in
the runs where the effect did appear.

---

## 1. The registered test

Primary metric: mean exact triplet F1 pooled over the four domains, H0b vs B2h,
both re-run on seeds 47-51, paired bootstrap over test sentences, 10,000
resamples.

| | exploratory, seeds 42-46 | confirmatory, seeds 47-51 |
|---|---:|---:|
| B2h | 69.24 | 69.45 |
| H0b | 69.65 | 69.37 |
| **diff** | **+0.41** | **-0.07** |
| 95% CI | - | [-0.54, +0.42] |
| p | - | 0.7679 |

The confirmatory interval **excludes the exploratory point estimate**. Per
domain, nothing survives and the one domain that had been significant loses most
of its effect:

| domain | exploratory diff | confirmatory diff | conf. p (Holm) |
|---|---:|---:|---:|
| 14res | +1.12 (Holm p = 0.029, **sig**) | +0.50 | 0.8100 |
| 14lap | +0.11 | -0.03 | 1.0000 |
| 16res | -0.38 | +0.09 | 1.0000 |
| 15res | +0.79 | -0.85 | 0.4864 |

Note 15res flips sign by 1.64 F1 between seed sets. For scale, B2h's own
four-domain average moves 69.24 -> 69.45 purely from changing seeds, so an
exploratory +0.41 selected as the best of three arms was never far outside what
seed noise produces on its own.

---

## 2. The mechanism was never there

Run before the confirmatory result, on the exploratory runs where H0b *did* look
positive (`scripts/aste_boundary_analysis.py`). If the listwise span objective
works, boundary errors fall and the sentiment decision stays put.

Sentiment stayed put, as predicted. Boundary errors did not move.

| % of gold spans partially matched (boundary wrong) | B2h | H0b |
|---|---:|---:|
| 14lap | 3.7 | 3.7 |
| 14res | 3.9 | 3.7 |
| 15res | 4.4 | 3.8 |
| 16res | 3.6 | 4.0 |

Exact-match rates are equally flat: aspect 85.7/89.1/85.6/91.2 for B2h against
85.4/89.1/85.8/90.7 for H0b. Polarity accuracy given both spans correct moves by
at most 0.4 points, which is the one thing the arm was *not* supposed to change
and correctly did not.

The per-gold-span transition matrix is the clearest statement of it. Matched by
domain, seed and sentence:

| domain | partial->exact | exact->partial | absent->exact | exact->absent | net |
|---|---:|---:|---:|---:|---:|
| 14lap | 25 | 22 | 123 | 117 | **+9** |
| 14res | 41 | 35 | 153 | 153 | **+6** |
| 15res | 36 | 29 | 112 | 108 | **+11** |
| 16res | 14 | 32 | 48 | 45 | **-15** |

Boundary corrections (`partial->exact`) are small and almost exactly cancelled
by boundary regressions. The large numbers are `absent<->exact` churn - spans
entering and leaving the emitted set - and those cancel too. Net movement in
correctly-bounded gold spans is +6 to +11 out of several thousand, and negative
on 16res.

**H0b did not reduce boundary errors.** Whatever produced its exploratory +0.41,
it was not the mechanism it was built on.

---

## 3. The headroom is real; H0b was not the way to it

The snap-boundary oracle takes B2h's own predictions, snaps every predicted span
that overlaps a gold span of its own role onto that gold span, dedupes (two
sloppy predictions collapsing onto one gold pair become one), and recomputes
ordinary exact triplet F1. Unlike relaxing the match to overlap, this is a decode
the model could in principle emit.

| domain | B2h | snap oracle | headroom | H0b gain (expl.) | recovery |
|---|---:|---:|---:|---:|---:|
| 14lap | 63.74 | 70.80 | +7.06 | +0.11 | 2% |
| 14res | 72.72 | 79.05 | +6.34 | +1.12 | 18% |
| 15res | 66.49 | 74.20 | +7.71 | +0.79 | 10% |
| 16res | 74.00 | 80.51 | +6.51 | -0.38 | -6% |
| **avg** | 69.24 | 76.14 | **+6.90** | +0.41 | **6%** |

Two things follow.

**The boundary opportunity is large and uniform: +6.9 F1 on average, and between
+6.3 and +7.7 on every single domain.** That is bigger than the gap from B2h to
MiniConGTS and about the same size as the gap to EPMEI. Span localisation is
where the remaining F1 is.

**The pre-specified explanation for 16res's sign flip is refuted.** The
hypothesis was that 16res would turn out to have little boundary headroom. It has
+6.51, essentially the same as everywhere else. 16res is not an outlier in
opportunity; H0b simply did not take any of it, there or anywhere.

---

## 4. What this costs and what it buys

Costs: the boundary-listwise objective is closed as a main-effect candidate. No
`w_boundary` sweep, no variant tuned on this test data.

Buys, and this is the larger half:

1. **A pre-registration caught a false positive before it became a claim.** The
   exploratory sweep produced +0.41 average with one Holm-significant domain,
   which is exactly the shape of result that gets written up. It does not
   replicate.
2. **A quantified target for the next arm.** +6.90 F1 of boundary headroom,
   uniform across domains, measured with an oracle that respects the real decode.
   Any future span-localisation arm can be scored as a fraction of it -
   `recovery ratio` in `scripts/aste_boundary_analysis.py` - rather than by
   whether it moved the average by more than seed noise.
3. **A negative control for the analysis itself.** H0b is now a known-null
   intervention with a full mechanism profile. A future arm that claims to fix
   boundaries has to beat H0b's transition matrix, not just its F1.

---

## Reproduction

```
GLINER_ABSA_DOMAINS=14res,14lap,16res,15res GLINER_ABSA_PRESETS=B2h,H0b \
GLINER_ABSA_SEEDS=47,48,49,50,51 python scripts/launch_h_sweep.py
python scripts/aste_confirm_test.py --seeds 47,48,49,50,51
python scripts/aste_boundary_analysis.py --arms B2h,H0b
```
