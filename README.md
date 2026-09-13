# RelAdapt-GLiNER2

**RelAdapt-GLiNER2: Relation-Aware Adaptation of GLiNER2 for Aspect Sentiment Triplet Extraction**

Xuan-Phuc Nguyen, Minh-Triet Nguyen, Duc-Hien Nguyen, Le-Tung-Duong Nguyen, Van-Thin Dang
University of Information Technology, VNU-HCM

This repository contains the code, stored predictions, and every experiment
reported in the paper (manuscript under review).

## Overview

GLiNER2 is a general-purpose, schema-driven span extractor. Aspect sentiment
triplet extraction (ASTE) instead requires tightly coupled aspect-opinion relation
decisions. We study how much of the general formulation transfers to ASTE, and
where the remaining errors come from.

- **GLiNER2-ASTE** is the baseline. On top of the GLiNER2 encoder it adds aspect
  and opinion span scorers, top-10 candidate pruning, a relation grid over
  {NONE, POS, NEU, NEG}, and a soft Sinkhorn assignment prior.
- **RelAdapt-GLiNER2** adds a label-specific biaffine aspect-opinion interaction,
  `b_ij^r = u_i^T W^r v_j + U^r [u_i; v_j] + c^r`, to the relation scorer.
- **Trajectory consensus** combines K independently trained models. It keeps a
  triplet only when at least `ceil(0.6 K)` of them predict it exactly.

Main findings:

1. **Biaffine interaction.** The gain is small but reproducible. With matched
   mini-batch order, it adds +1.53 dev F1 and +1.06 test F1 (95% CI
   [+0.43, +1.67]). An independent replication gives +1.22 dev and +1.12 test.
2. **Where errors arise.** Both gold spans survive candidate pruning for 95.3% of
   gold triplets. A relation oracle is worth +7.97 F1, against +6.33 for a boundary
   oracle.
3. **Where instability arises.** Across five trajectories, 75.4% of true-positive
   triplets are predicted by all five models, while 40.4% of false positives
   appear in only one. The TP-FP stability gap grows from spans (+0.94 aspect,
   +0.84 opinion) to pairs (+1.58) and triplets (+2.00).
4. **Consensus.** Exact-triplet consensus improves test F1 by +1.74 to +2.19 over
   the member models in three disjoint five-model groups.

These results are **not** a new ASTE state of the art. EPMEI reaches 75.26 average
F1, against 70.25 for RelAdapt-GLiNER2 and 72.29 for its best replicated five-model
consensus.

## Results

Test triplet F1 (%) on ASTE-Data-V2. Our single-model rows are means over five
seeds; "Members" is the mean F1 of the models being combined.

| | method | 14lap | 14res | 15res | 16res | avg | members | cost |
|---|---|---:|---:|---:|---:|---:|---:|---|
| reported | Span-ASTE | 59.38 | 71.85 | 63.27 | 70.26 | 66.19 | | |
| reported | BDTF | 61.74 | 74.35 | 66.12 | 72.27 | 68.62 | | |
| reported | MiniConGTS | 63.61 | 75.59 | 65.15 | 74.83 | 69.80 | | |
| reported | EPMEI | 69.64 | 80.98 | 73.01 | 77.41 | **75.26** | | |
| ours | GLiNER2-ASTE | 63.64 | 73.46 | 65.94 | 73.73 | 69.19 | | 1x |
| ours | RelAdapt-GLiNER2 | 64.52 | 74.10 | 67.79 | 74.59 | 70.25 | | 1x |
| ours | + consensus-5, group A* | 65.71 | 75.79 | 70.30 | 76.15 | 71.99 | 70.25 | 5x |
| ours | + consensus-5, group B | 66.60 | 75.20 | 70.67 | 76.12 | 72.15 | 69.95 | 5x |
| ours | + consensus-5, group C | 66.17 | 75.30 | 71.27 | 76.44 | 72.29 | 70.45 | 5x |
| ours | + consensus-10, B+C | 66.48 | 75.88 | 72.24 | 76.88 | 72.87 | 70.20 | 10x |

\* The consensus ratio 0.6 was selected on group A development predictions, then
frozen for groups B and C.

**Ablations** (mean dev / test F1):

| comparison | baseline | with change |
|---|---|---|
| scorer screen, deeper MLP + features (seeds 42-46) | 70.76 / 69.38 | 70.30 / 69.53 |
| scorer screen, + biaffine | 70.76 / 69.38 | 71.33 / 69.99 |
| scorer screen, + biaffine + features | 70.76 / 69.38 | 71.33 / 70.31 |
| independent replication (seeds 52-56) | 70.19 / 69.44 | 71.41 / 70.56 |
| canonical matched order (seeds 62-66) | 69.93 / 69.19 | 71.46 / 70.25 |
| generic span-pair head -> + Sinkhorn prior | 61.25 / 60.47 | 68.51 / 67.32 |
| + Sinkhorn prior -> + hard-negative weighting | 68.51 / 67.32 | 70.18 / 69.24 |

- **Sinkhorn prior.** It is kept because it raises precision substantially. It does
  lower recall on the 38.4% of test triplets that share an aspect or an opinion
  with another triplet.
- **Negative results.** None of these improved the common baseline: polarity-aware
  training, a boundary loss (did not replicate), overlap-aware span ranking,
  listwise and NULL-focused objectives, and type-conditioned pair features.

## Limitations

- **Test exposure.** The standard test splits were inspected during early
  exploration, so test numbers are indicative. The biaffine configuration and the
  consensus ratio were chosen on development data, then replicated.
- **Inference cost.** Consensus costs K times single-model inference, and the
  benefit of going from 5 to 10 models is about +0.6-0.7 F1.
- **Scope.** One base-size encoder and four English datasets.

## Repository layout

```
Data/ASTE/            ASTE-Data-V2 (14lap, 14res, 15res, 16res; train/dev/test)
research/
  aste_full.py          model and presets: span encoder, role scorers, relation scorers
  aste_full_train.py    training loop, decoding, evaluation
  aste_data.py          ASTE-Data-V2 loader
  aste_polarity.py      polarity and boundary objectives (negative results)
  aste_ranking.py       overlap-aware span ranking and rank diagnostics
  aste_relation.py      listwise / NULL relation objectives and diagnostics
  aste_stats.py         paired bootstrap and significance tests
  modal_research.py     Modal app that runs training on GPU
scripts/
  launch_aste.py, launch_h_sweep.py, monitor_aste.py, collect_runs.py
  aste_consensus.py            cross-seed stability and consensus decoding
  aste_transition_analysis.py  where the biaffine gain comes from
  aste_error_budget.py         relation and boundary oracles
  aste_confirm_test.py         pre-registered paired confirmatory test
  aste_boundary_analysis.py    boundary-error rates
  aste_arm_decomposition.py    linking and span-quality decomposition
results/              one metrics.json per run, plus pre-registrations
analysis/             scripts that recompute the paper's tables, statistics, and figure
```

### Experiments and run names

Each run is stored at `results/aste_<domain>/<arm>/seed_<n>/metrics.json`. The
configuration of an arm is the preset of the same name in `research/aste_full.py`;
a suffix such as `-cf` only names the seed group.

| paper section | arms | seeds |
|---|---|---|
| main results, stability, consensus | `B2h-cf` (GLiNER2-ASTE), `R1bm-cf` (RelAdapt-GLiNER2), `R1bm-cs` | 62-66, 67-76 |
| relation-scorer screen | `B2h-ex3`, `R1m`, `R1b`, `R1bm` | 42-46 |
| independent replication | `B2h-c2`, `R1bm-c2` | 52-56 |
| building the baseline | `A0` (generic span-pair head), `B2` (+Sinkhorn prior), `B2h` (+hard negatives) | 42-46 |
| negative results | `H1`, `H3`, `L1`, `L1a`, `M1`-`M3` | 42-46 |
| negative results | `H0b`, `K1`, `K1e`, `J1`, `J2`, with `B2h` as control | 47-51 |

Each `metrics.json` records:

- the full config and parameter counts;
- dev/test P/R/F1 and a decoder funnel;
- per-sentence gold and predicted triplets;
- threshold curves;
- `batch_order_hash` for runs after the data-order fix.

Other material in `results/`:

- pre-registrations: `results/aste_{canonical,consensus,r1bm,h0b}_confirm/`;
- the zero-shot span probe: `results/aste_baseline/`.

## Reproducing

### Analyses (CPU only, from stored predictions)

```bash
pip install -r requirements.txt
python scripts/aste_consensus.py --arm R1bm-cf --seeds 62,63,64,65,66 --split dev
python scripts/aste_transition_analysis.py --a B2h-cf --b R1bm-cf
python analysis/paper_numbers.py dev test   # per-domain single-model and consensus tables
python analysis/paper_stats.py              # bootstrap CIs, ensemble-size curve, funnel
python analysis/sinkhorn_degree.py          # relation degree and shared-span recall
python analysis/make_figures.py             # figures/stability-analysis.pdf
```

### Training (Modal GPU)

Runs execute in the Modal image defined by `research/modal_research.py`, which pins
`gliner2[local]==2.0.0`. From the repository root:

```bash
modal deploy research/modal_research.py
GLINER_ABSA_DOMAIN=15res GLINER_ABSA_PRESETS=B2h,R1bm \
GLINER_ABSA_SEEDS=62,63,64,65,66 GLINER_ABSA_SAVE_SUFFIX=-cf \
python scripts/launch_aste.py
GLINER_ABSA_OUT_DIR=results/aste_15res python scripts/monitor_aste.py
```

- `GLINER_ABSA_MODAL_PROFILE` selects a Modal profile. If it is unset, the active
  default profile is used.
- `scripts/launch_h_sweep.py` shards a sweep across the profiles listed in
  `GLINER_ABSA_PROFILES`.
- `GLINER_ABSA_GPU` selects the GPU type; the default is `A100`.
- On Windows, set `PYTHONUTF8=1` before calling the Modal CLI.

## Data

`Data/ASTE` is ASTE-Data-V2 from Xu et al., *Position-Aware Tagging for Aspect
Sentiment Triplet Extraction* (EMNLP 2020). It is derived from the SemEval
2014-2016 ABSA tasks. Please cite the original authors when using it.

## Citation

```bibtex
@misc{nguyen2026reladaptgliner2,
  title  = {{RelAdapt-GLiNER2}: Relation-Aware Adaptation of {GLiNER2} for Aspect Sentiment Triplet Extraction},
  author = {Nguyen, Xuan-Phuc and Nguyen, Minh-Triet and Nguyen, Duc-Hien and Nguyen, Le-Tung-Duong and Dang, Van-Thin},
  year   = {2026},
  note   = {Manuscript}
}
```
