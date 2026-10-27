# RelAdapt

**RelAdapt: Schema-Driven Span Matching with a Bidirectional Transformer Encoder for Aspect Sentiment Triplet Extraction**

Nguyen Xuan Phuc, Pham Gia An, Nguyen Minh Triet, Nguyen Duc Hien, Nguyen Le Tung Duong, Dang Van Thin
University of Information Technology, Vietnam National University Ho Chi Minh City

This repository contains the code, the stored predictions of every run, and the
scripts that recompute every number in the paper (manuscript under review).

## Overview

Recent work treats aspect sentiment triplet extraction (ASTE) as text generation:
an encoder–decoder model emits a linearised triplet string that is parsed back into
spans. RelAdapt instead solves ASTE as **schema-driven matching** on a bidirectional
Transformer encoder, in a single forward pass and without any decoding.

- **Role matching.** Every span of up to 8 tokens is matched against learned
  `aspect` and `opinion` slots; the 10 best spans per role are kept.
- **Relation matching.** Every aspect–opinion candidate pair is matched against the
  relation slots `{NONE, POS, NEU, NEG}` by a concatenation matcher plus a
  label-specific biaffine term `b_ij^r = u_i^T W^r v_j + U^r [u_i; v_j] + c^r`.
- **Sinkhorn assignment prior** with dustbins softly encourages one-to-one pairing,
  and **hard-negative weighting** focuses the relation loss on pairs that share a
  gold span.
- **Trajectory consensus** keeps a triplet when at least `ceil(0.6 K)` of `K`
  independently trained models predict it exactly.

Every predicted span is an input span by construction, and the output is always a
well-formed set of triplets. The encoder is interchangeable; the main backbone is
DeBERTa-v3-base.

## Results

Test triplet F1 (%) on ASTE-Data-V2 (DeBERTa-v3-base, five seeds per row, matched
mini-batch order across all configurations).

| method | 14lap | 14res | 15res | 16res | avg |
|---|---:|---:|---:|---:|---:|
| MvP (T5-base, five views) | 63.33 | 74.05 | 65.89 | 73.48 | 69.19 |
| MPGM (T5) | 65.32 | 76.80 | 68.75 | 75.49 | 71.59 |
| DGSEP S2 (T5) | 67.78 | 75.63 | 68.58 | 74.66 | 71.66 |
| Concatenation matcher | 64.41 | 74.14 | 65.48 | 73.61 | 69.41 |
| **RelAdapt** (single model) | 66.50 | 74.84 | 67.84 | 74.96 | **71.04** |
| RelAdapt + consensus-5, group A | 68.68 | 76.71 | 70.18 | 77.26 | 73.21 |
| RelAdapt + consensus-5, group B | 67.79 | 76.97 | 70.09 | 76.91 | 72.94 |
| RelAdapt + consensus-5, group C | 68.76 | 76.98 | 70.46 | 77.44 | 73.41 |
| RelAdapt + consensus-10, B ∪ C | 68.41 | 77.32 | 71.31 | 77.84 | **73.72** |

Generative rows are as reported in their papers. The paper compares against 20
published systems.

**The biaffine matcher improves every encoder** (test F1, five seeds,
paired-bootstrap 95% CI):

| encoder | concat. matcher | RelAdapt | Δ |
|---|---:|---:|---|
| BERT-base-uncased | 64.00 | 65.35 | +1.36 [+0.63, +2.08] |
| RoBERTa-base | 67.72 | 69.53 | +1.81 [+1.14, +2.49] |
| DeBERTa-v3-base | 69.41 | 71.04 | +1.63 [+0.91, +2.36] |
| DeBERTa-v3-large | 67.33 | 70.80 | +3.47 [+2.80, +4.14] |

**Inference cost** (1,468 test sentences, NVIDIA A10G, ms per sentence):

| system | batch 1 | batch 32 |
|---|---:|---:|
| RelAdapt, DeBERTa-v3-base | 19.5 | 1.8 |
| RelAdapt, consensus-5 | 97.7 | 8.95 |
| T5-base generation, GAS style | 202.6 | 20.1 |
| T5-base generation, MvP style (5 views) | 1,563.5 | 163.3 |

**Analyses.** For 96.3% of the gold triplets both spans reach the relation matcher;
relation decisions carry most of the remaining errors and are the least stable part
across training runs (a true positive is predicted by 4.45 of five models on
average, a false positive by 2.42). The biaffine matcher works through selectivity
(2,020 false positives removed, 1,443 added) and recovers the rare neutral polarity
without any polarity-specific objective.

## Repository layout

```
Data/ASTE/            ASTE-Data-V2 (14lap, 14res, 15res, 16res; train/dev/test)
research/
  aste_full.py          model and presets: span encoder, role matching, relation matchers
  aste_full_train.py    training loop, decoding, evaluation (any Hugging Face encoder)
  aste_data.py          ASTE-Data-V2 loader
  aste_speed.py         inference-cost benchmark against T5-base generation
  aste_polarity.py      polarity and boundary objectives (alternative designs)
  aste_ranking.py       overlap-aware span ranking and rank diagnostics
  aste_relation.py      listwise / NULL relation objectives and diagnostics
  aste_stats.py         paired bootstrap and significance tests
  modal_research.py     Modal app (GPU, or CPU with GLINER_ABSA_GPU=cpu)
scripts/
  launch_aste.py, collect_runs.py, monitor_aste.py
  backbone_sweep.py     the backbone study: launch and collect every configuration
  collect_all.py        pull finished runs and the speed benchmark back from Modal
  run_speed.py          run the inference-cost benchmark
  aste_consensus.py, aste_error_budget.py, aste_transition_analysis.py, ...
analysis/
  backbone_paper.py     every number and the stability figure of the paper
  encoder_control*.py   encoder comparison under matched mini-batch order
results/              one metrics.json per run
```

### Run names

Each run is stored at `results/aste_<domain>/<arm>/seed_<n>/metrics.json`. The arm
name is the preset in `research/aste_full.py` followed by an encoder suffix.

| preset | configuration |
|---|---|
| `A0` | generic span-pair classifier |
| `B2` | + Sinkhorn assignment prior |
| `B2h` | + hard-negative weighting (concatenation matcher) |
| `R1m` | + deeper matcher and pair features |
| `R1b` | + label-specific biaffine |
| `R1bm` | + biaffine and pair features (**RelAdapt**) |

| suffix | encoder |
|---|---|
| `-enc-dv3` | `microsoft/deberta-v3-base` (main backbone) |
| `-enc-dv3l` | `microsoft/deberta-v3-large` (encoder learning rate 1e-5) |
| `-enc-rob` | `FacebookAI/roberta-base` |
| `-enc-bert` | `bert-base-uncased` |
| `-enc-gl2` | DeBERTa-v3 initialised from `fastino/gliner2-base-v1` |

Seed groups: A = 62–66 (all configurations), B = 67–71 and C = 72–76 (RelAdapt).
All runs with the same seed share an identical mini-batch order, recorded as
`batch_order_hash`. Arms without an `-enc-` suffix are development experiments on
the GLiNER2-initialised encoder (the alternative designs of the paper).

## Reproducing

### Analyses (CPU only, from stored predictions)

```bash
pip install -r requirements.txt
python analysis/backbone_paper.py          # all tables, bootstrap CIs, figure
python analysis/encoder_control_stats.py   # encoder comparison
python scripts/aste_error_budget.py --arm R1bm-enc-dv3 --seeds 62,63,64,65,66 --split dev --json /tmp/budget.json
```

### Training (Modal)

```bash
modal deploy research/modal_research.py                      # GPU (GLINER_ABSA_GPU=A10G)
GLINER_ABSA_GPU=cpu modal deploy research/modal_research.py  # or a CPU-only deployment
GLINER_ABSA_DOMAIN=15res GLINER_ABSA_PRESETS=B2h,R1bm \
GLINER_ABSA_MODEL=microsoft/deberta-v3-base GLINER_ABSA_SAVE_SUFFIX=-enc-dv3 \
GLINER_ABSA_SEEDS=62,63,64,65,66 GLINER_ABSA_RNG_NEUTRAL=1 \
python scripts/launch_aste.py
GLINER_ABSA_DOMAIN=15res python scripts/collect_runs.py
```

- `GLINER_ABSA_MODAL_PROFILE` selects a Modal profile; `backbone_sweep.py` spreads
  the whole study over the profiles in `GLINER_ABSA_GPU_PROFILES`.
- `GLINER_ABSA_RNG_NEUTRAL=1` keeps the mini-batch order identical across encoders.
- On Windows, set `PYTHONUTF8=1` before calling the Modal CLI.

## Data

`Data/ASTE` is ASTE-Data-V2 from Xu et al., *Position-Aware Tagging for Aspect
Sentiment Triplet Extraction* (EMNLP 2020), derived from the SemEval 2014–2016 ABSA
tasks. Please cite the original authors when using it.

## Citation

```bibtex
@misc{nguyen2026reladapt,
  title  = {{RelAdapt}: Schema-Driven Span Matching with a Bidirectional Transformer Encoder for Aspect Sentiment Triplet Extraction},
  author = {Nguyen, Xuan Phuc and Pham, Gia An and Nguyen, Minh Triet and Nguyen, Duc Hien and Nguyen, Le Tung Duong and Dang, Van Thin},
  year   = {2026},
  note   = {Manuscript under review}
}
```
