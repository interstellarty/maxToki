# Phase 4 — GSN KO in Cardiac Fibroblasts: Results

**Run:** 2026-04-22T11:06:03  
**Script:** `phase4_gsn_heart.py`  
**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  
**Data:** TSP1_30 heart — cardiac fibroblasts, 238 young / 534 old  
**Age split:** young ≤ 45, old ≥ 60  
**Pairs:** 1 young anchor × 500 old cells = 500 pairs  
**Specs:** 1 baseline + 1 GSN KO + 50 null KOs = 52

## Summary

| gene | effect_rate | n_effective | mean_delta | std_delta | null_mean | null_std | z_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSN | 0.9800 | 490 | -0.0006 | 0.0060 | 0.0006 | 0.0015 | -0.8297 |

## Interpretation

**GSN KO z = -0.83** — opposite direction from the published hit. Either (a) the paper used the fine-tuned TimeBetweenCells head and pretraining LL does not capture the signal, (b) anchor choice or age split is too different, or (c) the effect is fragile. Replication **not supported**.


## Null distribution (50 random-gene KOs, effective only)

- Effective null genes: 41

- Null mean delta: +0.0006

- Null std: 0.0015

## Provenance

Outputs in `/ptmp/artfi/heart_screen_phase4/screen_gsn_v1/`:

- `scored_results.csv`
- `null_per_gene.csv`

Regenerate with `python phase4_gsn_heart.py`.
