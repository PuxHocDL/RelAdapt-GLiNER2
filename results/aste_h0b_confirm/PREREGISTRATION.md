# Pre-registration: H0b vs B2h, confirmatory

Written **before** any run on these seeds existed. Timestamped by the launch
manifests in this directory; the exploratory runs that motivated it are on seeds
42-46 and are not reused here for anything except the hypothesis.

## Why this is a separate experiment

H0b was discovered in a 3-arm x 4-domain exploratory sweep on seeds 42-46 in
which H1 and H3 were also tested. It came out at +0.41 average triplet F1 over
B2h, positive on 3 of 4 domains, significant after Holm on 1 of 4 (14res,
+1.12, p=0.029). Reporting that as a result would be reporting the best of three
arms selected on the same data that measured it. This run exists to test the
surviving hypothesis on data that had no part in generating it.

## Hypothesis

> Replacing the per-span independent sigmoid span objective with a listwise
> softmax over end positions per start position (`use_boundary_listwise`,
> `w_boundary = 0.5`) improves exact triplet F1 on ASTE-Data-V2 relative to B2h.

Directional: H0b > B2h. One-sided in intent, reported two-sided.

## The single pre-specified comparison

**Primary.** Mean exact triplet F1 at the untuned threshold, averaged over the
four domains, H0b vs B2h. Paired bootstrap over test sentences, 10,000
resamples, pooled across domains and seeds. alpha = 0.05. **This is the only
comparison the conclusion rests on.**

**Secondary, exploratory, no multiplicity budget spent on the primary claim.**
Per-domain diffs with Holm correction across the four domains; per-class
polarity F1; the dev-tuned-threshold number.

## Locked before launch

| | |
|---|---|
| arms | exactly two: `B2h`, `H0b`. Nothing else runs on these seeds. |
| seeds | 47, 48, 49, 50, 51 - disjoint from the exploratory 42-46 |
| domains | 14lap, 14res, 15res, 16res |
| model | `fastino/gliner2-base-v1` |
| epochs | 20 |
| batch size | 8 |
| encoder LR / head LR | 2e-5 / 1e-4 |
| warmup | 0.2, OneCycle cosine |
| `w_boundary` | 0.5 (the exploratory value; **not** re-tuned) |
| everything else | `AsteConfig.preset` defaults, untouched |
| GPU | A10G |

No hyperparameter is tuned on these seeds, for either arm. `w_boundary` keeps
the first value ever tried. The control is re-run on the same fresh seeds rather
than reusing the 42-46 B2h runs, so the pairing is exact.

## Decision rule, fixed in advance

- **Confirmed** if the pooled primary diff is positive and its bootstrap p < 0.05.
- **Not confirmed** otherwise. In that case H0b is reported as an exploratory
  finding that did not replicate, and no further boundary variant is tuned on
  ASTE-Data-V2 test data.

A per-domain sign flip does **not** by itself refute the hypothesis - the
primary metric is the average - but it is reported, and the boundary-oracle
headroom per domain (`scripts/aste_boundary_analysis.py`) is the pre-specified
explanation to check against it.

## What would make this fail honestly

16res was negative in exploration (-0.38). If its boundary-oracle headroom is
small, that is an explanation; if its headroom is large and H0b still loses
there, the mechanism claim is wrong even if the average holds.
