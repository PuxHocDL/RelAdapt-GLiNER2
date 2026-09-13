# Published ASTE results on ASTE-Data-V2

**Provenance.** Every number on this page was supplied by the user, together with
the sources they checked. They have **not** been independently re-verified here —
treat them as a working reference for positioning our own runs, and re-check
against the primary paper before quoting any of them in a write-up.

Our data splits were verified against the V2 statistics below and match exactly
on all 24 counts (see `baseline.md`), so these numbers are directly comparable to
ours in terms of data.

## 1. Dataset size (ASTE-Data-V2)

| Dataset | Train sent. | Dev | Test | Train tri. | Dev tri. | Test tri. |
|---|---:|---:|---:|---:|---:|---:|
| 14lap | 906 | 219 | 328 | 1,460 | 346 | 543 |
| 14res | 1,266 | 310 | 492 | 2,338 | 577 | 994 |
| 15res | 605 | 148 | 322 | 1,013 | 249 | 485 |
| 16res | 857 | 210 | 326 | 1,394 | 339 | 514 |

## 2. Triplet F1 across systems

| Model | Year | 14lap | 14res | 15res | 16res | Avg |
|---|---|---:|---:|---:|---:|---:|
| Peng two-stage | 2020 | 42.87 | 51.46 | 52.32 | 54.21 | 50.22 |
| JET-BERT | 2020 | 51.04 | 62.40 | 57.53 | 63.83 | 58.70 |
| GTS-BERT | 2020 | 54.36 | 67.50 | 60.15 | 67.93 | 62.49 |
| Dual-MRC | 2021 | 55.58 | 70.32 | 57.21 | 67.40 | 62.63 |
| Span-ASTE | 2021 | 59.38 | 71.85 | 63.27 | 70.26 | 66.19 |
| EMC-GCN | 2022 | 58.81 | 71.78 | 61.93 | 68.33 | 65.21 |
| SSJE | 2022 | 60.41 | 72.26 | 65.05 | 71.38 | 67.28 |
| BDTF | 2022 | 61.74 | 74.35 | 66.12 | 72.27 | 68.62 |
| DGCNAP | 2023 | 57.57 | 70.72 | 61.19 | 69.58 | 64.77 |
| STAGE-1D | 2023 | 61.49 | 73.58 | 64.37 | 73.45 | 68.22 |
| STAGE-2D | 2023 | 61.88 | 73.61 | 64.94 | 72.75 | 68.30 |
| STAGE-3D | 2023 | 61.58 | 73.76 | 64.79 | 73.24 | 68.34 |
| PBLUN | 2023/24 | 62.51 | 75.03 | 66.88 | 72.89 | 69.33 |
| MiniConGTS | 2024 | 63.61 | 75.59 | 65.15 | 74.83 | 69.80 |
| MVLF-SL | 2025 | 63.82 | 74.92 | 68.32 | 74.01 | 70.27 |
| MPGM | 2025 | 65.32 | 76.80 | 68.75 | 75.49 | 71.59 |
| GAC | 2025 | 66.47 | 76.01 | 69.04 | 76.25 | 71.94 |
| DGSEP-S1 | 2026 | 66.00 | 75.39 | 68.15 | 75.54 | 71.27 |
| DGSEP-S2 | 2026 | 67.78 | 75.63 | 68.58 | 74.66 | 71.66 |
| **EPMEI** | SIGIR 2024 | **69.64** | **80.98** | 73.01 | **77.41** | **75.26** |
| DESS-V2-xxLarge* | 2025 preprint | 69.08 | 79.98 | *74.22* | 77.16 | 75.11 |

\* DESS is an arXiv preprint, not peer-reviewed. Do not place it level with
peer-reviewed SOTA without saying so.

## 3. Precision / Recall / F1 for the strongest systems

| Model | Dataset | P | R | F1 |
|---|---|---:|---:|---:|
| EPMEI | 14lap | 73.06 | 66.55 | 69.64 |
| | 14res | 86.21 | 76.37 | 80.98 |
| | 15res | 76.36 | 70.10 | 73.01 |
| | 16res | 82.93 | 72.60 | 77.41 |
| DESS-V2-xxLarge* | 14lap | 72.88 | 65.65 | 69.08 |
| | 14res | 80.78 | 79.20 | 79.98 |
| | 15res | 70.35 | 75.16 | 74.22 |
| | 16res | 80.72 | 77.33 | 77.16 |
| DGSEP-S2 | 14lap | 72.33 | 63.77 | 67.78 |
| | 14res | 77.51 | 73.84 | 75.63 |
| | 15res | 70.68 | 66.60 | 68.58 |
| | 16res | 75.40 | 73.93 | 74.66 |
| MiniConGTS | 14lap | 66.82 | 60.68 | 63.61 |
| | 14res | 76.10 | 75.08 | 75.59 |
| | 15res | 66.50 | 63.86 | 65.15 |
| | 16res | 75.52 | 74.14 | 74.83 |
| PBLUN | 14lap | 66.84 | 58.71 | 62.51 |
| | 14res | 76.97 | 73.20 | 75.03 |
| | 15res | 69.64 | 64.33 | 66.88 |
| | 16res | 72.11 | 73.74 | 72.89 |
| BDTF | 14lap | 68.94 | 55.97 | 61.74 |
| | 14res | 75.53 | 73.24 | 74.35 |
| | 15res | 68.76 | 63.71 | 66.12 |
| | 16res | 71.44 | 73.13 | 72.27 |

## 4. Classic baselines, P/R/F1 (from the PBLUN paper's Table 2)

| Model | 14lap P/R/F1 | 14res P/R/F1 | 15res P/R/F1 | 16res P/R/F1 |
|---|---|---|---|---|
| Peng two-stage | 37.38 / 50.38 / 42.87 | 43.24 / 63.66 / 51.46 | 48.07 / 57.51 / 52.32 | 46.96 / 64.24 / 54.21 |
| GTS | 57.82 / 51.32 / 54.36 | 67.76 / 67.29 / 67.50 | 62.59 / 57.94 / 60.15 | 66.08 / 66.91 / 67.93 |
| Span-ASTE | 63.44 / 55.84 / 59.38 | 72.89 / 70.89 / 71.85 | 62.18 / 64.45 / 63.27 | 69.45 / 71.17 / 70.26 |
| SSJE | 67.43 / 54.71 / 60.41 | 73.12 / 71.43 / 72.26 | 63.94 / 66.17 / 65.05 | 70.82 / 72.00 / 71.38 |
| EMC-GCN | 61.70 / 56.26 / 58.81 | 71.21 / 72.39 / 71.78 | 61.54 / 62.47 / 61.93 | 65.62 / 71.30 / 68.33 |
| BDTF | 68.94 / 55.97 / 61.74 | 75.53 / 73.24 / 74.35 | 68.76 / 63.71 / 66.12 | 71.44 / 73.13 / 72.27 |
| PBLUN | 66.84 / 58.71 / 62.51 | 76.97 / 73.20 / 75.03 | 69.64 / 64.33 / 66.88 | 72.11 / 73.74 / 72.89 |

## 5. Prompting-only LLM baselines (from MiniConGTS, Table 1)

| Model / prompting | 14lap | 14res | 15res | 16res |
|---|---:|---:|---:|---:|
| GPT-3.5 zero-shot | 34.69 | 49.48 | 43.02 | 47.22 |
| GPT-3.5 few-shot | 44.35 | 57.55 | 51.39 | 59.45 |
| GPT-3.5 CoT | 34.71 | 53.24 | 46.57 | 52.10 |
| GPT-3.5 CoT + few-shot | 37.57 | 53.85 | 46.02 | 54.86 |
| GPT-4o zero-shot | 19.90 | 35.37 | 32.05 | 36.80 |
| GPT-4o few-shot | 42.80 | 59.55 | 51.95 | 60.63 |
| GPT-4o CoT | 31.46 | 46.49 | 40.10 | 46.79 |
| GPT-4o CoT + few-shot | 34.40 | 52.54 | 42.47 | 49.45 |
| MiniConGTS (supervised) | 63.61 | 75.59 | 65.15 | 74.83 |

Exact-triplet ASTE is hard for prompting-only LLMs: the best GPT-4o setting sits
roughly 15-20 F1 below a supervised span model.

## 6. Best verified numbers per dataset

| Dataset | Best | Model | Status |
|---|---:|---|---|
| 14lap | 69.64 | EPMEI | SIGIR 2024, peer-reviewed |
| 14res | 80.98 | EPMEI | SIGIR 2024, peer-reviewed |
| 15res | 74.22 | DESS-V2-xxLarge | arXiv preprint |
| 15res (peer-reviewed) | 73.01 | EPMEI | SIGIR 2024 |
| 16res | 77.41 | EPMEI | SIGIR 2024, peer-reviewed |

Sources cited by the user: EPMEI (SIGIR 2024, DOI 10.1145/3626772.3657734),
DGSEP (Expert Systems with Applications, Vol. 299, March 2026), MiniConGTS
(EMNLP 2024), PBLUN Table 2 for the classic baselines.
