# Phase 4b — GSN Cardiac-Fibroblast Replication: Multi-anchor + soft-OE

**Run:** 2026-04-22T16:48:49  
**Script:** `phase4b_gsn_heart_ensemble.py`  
**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  
**Data:** TSP1_30 heart cardiac fibroblasts — reuses Phase 4 tokenization  
**Pairs:** 10 anchors × 200 old cells = 2000  
**Specs:** 1 baseline + 2 target + 50 null-KO + 50 null-sOE = 103  
**Prompts scored:** 206000

## Summary

| arm | n_cells | mean_delta_ens | null_mean | null_std | n_null_genes | z_score |
| --- | --- | --- | --- | --- | --- | --- |
| KO:GSN | 198 | -0.0001 | 0.0003 | 0.0015 | 40 | -0.3123 |
| sOE50:GSN | 2 | 0.0020 | 0.0185 | 0.0103 | 50 | -1.6050 |

## Interpretation

**KO:GSN: z = -0.31.** Opposite direction from a rejuvenation hit, or effectively null.

**sOE50:GSN: z = -1.60.** Opposite direction from a rejuvenation hit, or effectively null.


## Comparison to Phase 4 (single-anchor)

Phase 4 (single anchor, N=500 old cells): GSN KO z = **−0.83**, mean_delta = −0.0006. Phase 4b multi-anchor averages across 10 young anchors per old cell, which should suppress anchor-identity noise if it was masking a real signal. If the KO arm moves from z ≈ −1 to z ≥ +1, anchor noise was the bottleneck. If it stays null, the pretraining-only LL readout genuinely does not resolve GSN — next step is to evaluate with the fine-tuned TimeBetweenCells regression head.

## Provenance

Outputs in `/ptmp/artfi/heart_screen_phase4b/screen_gsn_ensemble_v1/`:

- `scored_results.csv`
- `per_old_ensemble.csv`

Regenerate with `python phase4b_gsn_heart_ensemble.py`.
