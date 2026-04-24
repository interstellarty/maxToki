# Phase 5 — Liver Fibrosis KO Screen: Results (Ramachandran 2019)

**Run:** 2026-04-24T10:45:23  
**Script:** `phase5_liver_fibrosis.py`  
**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  
**Data:** Ramachandran 2019 liver — 704 activated (cirrhotic) / 1504 quiescent (healthy) HSCs  
**Pairs:** 1 quiescent anchor × 500 activated = 500  
**Specs:** 1 baseline + 17 antifibrotic + 4 pro-fibrotic neg-ctrl + 50 null = 72  
**Prompts scored:** 36000

## Null distribution (50 random-gene KOs, effective only)

- Effective null genes: 32

- Null mean delta: +0.0007

- Null std: 0.0016

## KO effect rates (fraction of activated HSCs where KO dropped a token)

Low effect rate → gene rarely in top-2000 tokens → z-score rests on few cells.

| gene_symbol | gene_category | effect_rate |
| --- | --- | --- |
| TIMP1 | antifibrotic | 0.91 |
| ACTA2 | antifibrotic | 0.84 |
| PDGFRB | antifibrotic | 0.62 |
| COL1A1 | antifibrotic | 0.53 |
| COL1A2 | antifibrotic | 0.52 |
| COL3A1 | antifibrotic | 0.52 |
| TIMP2 | antifibrotic | 0.35 |
| CCN2 | antifibrotic | 0.33 |
| FN1 | antifibrotic | 0.29 |
| PDGFRA | antifibrotic | 0.20 |
| MMP2 | pro_fibrotic_neg_ctrl | 0.18 |
| TGFB1 | antifibrotic | 0.18 |
| TGFBR2 | antifibrotic | 0.16 |
| SMAD2 | antifibrotic | 0.10 |
| LOXL2 | antifibrotic | 0.07 |
| SMAD7 | pro_fibrotic_neg_ctrl | 0.06 |
| LOX | antifibrotic | 0.04 |
| SMAD3 | antifibrotic | 0.03 |
| TGFBR1 | antifibrotic | 0.03 |
| MMP9 | pro_fibrotic_neg_ctrl | 0.00 |
| MMP1 | pro_fibrotic_neg_ctrl | 0.00 |

## Z-scored per-target results

`delta_vs_baseline > 0` = perturbed activated HSC looks more like the quiescent anchor. `z >= +2.0` is notable. Antifibrotic KOs should score positive; pro-fibrotic neg-ctrl KOs should not.

| gene_symbol | gene_category | spec_class | mean_delta | n_effective | null_mean | null_std | z_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CCN2 | antifibrotic | antifibrotic | 0.0013 | 165 | 0.0007 | 0.0016 | 0.3677 |
| ACTA2 | antifibrotic | antifibrotic | 0.0011 | 418 | 0.0007 | 0.0016 | 0.2821 |
| TIMP1 | antifibrotic | antifibrotic | 0.0009 | 453 | 0.0007 | 0.0016 | 0.1402 |
| SMAD7 | pro_fibrotic_neg_ctrl | neg_ctrl | 0.0008 | 29 | 0.0007 | 0.0016 | 0.0981 |
| TGFBR1 | antifibrotic | antifibrotic | 0.0003 | 15 | 0.0007 | 0.0016 | -0.2473 |
| PDGFRA | antifibrotic | antifibrotic | 0.0001 | 101 | 0.0007 | 0.0016 | -0.3301 |
| TGFB1 | antifibrotic | antifibrotic | 0.0001 | 88 | 0.0007 | 0.0016 | -0.3414 |
| PDGFRB | antifibrotic | antifibrotic | 0.0001 | 308 | 0.0007 | 0.0016 | -0.3667 |
| SMAD2 | antifibrotic | antifibrotic | 0.0001 | 49 | 0.0007 | 0.0016 | -0.3742 |
| LOX | antifibrotic | antifibrotic | 0.0001 | 19 | 0.0007 | 0.0016 | -0.3774 |
| TIMP2 | antifibrotic | antifibrotic | -0.0000 | 173 | 0.0007 | 0.0016 | -0.4238 |
| FN1 | antifibrotic | antifibrotic | -0.0000 | 147 | 0.0007 | 0.0016 | -0.4401 |
| COL3A1 | antifibrotic | antifibrotic | -0.0002 | 258 | 0.0007 | 0.0016 | -0.5399 |
| TGFBR2 | antifibrotic | antifibrotic | -0.0003 | 80 | 0.0007 | 0.0016 | -0.6094 |
| COL1A1 | antifibrotic | antifibrotic | -0.0004 | 264 | 0.0007 | 0.0016 | -0.6759 |
| LOXL2 | antifibrotic | antifibrotic | -0.0005 | 37 | 0.0007 | 0.0016 | -0.7048 |
| SMAD3 | antifibrotic | antifibrotic | -0.0005 | 17 | 0.0007 | 0.0016 | -0.7192 |
| MMP2 | pro_fibrotic_neg_ctrl | neg_ctrl | -0.0012 | 89 | 0.0007 | 0.0016 | -1.1220 |
| COL1A2 | antifibrotic | antifibrotic | -0.0014 | 260 | 0.0007 | 0.0016 | -1.2619 |

## Key findings

**No antifibrotic KO passed z >= +2.0.** Either the LL-as-distance-to-quiescent framing is underpowered at this scale (next step: Phase 5b with multi-anchor ensembling, Phase 4b's trick), or the identity-saturation issue from Phase 2b extends to disease-state as well.

No pro-fibrotic neg-ctrl KO passed |z| >= 2.0. Negative controls behave as expected.

## Provenance

Outputs in `/ptmp/artfi/liver_fibrosis_phase5/screen_hsc_fibrosis_v1/`:

- `scored_results.csv`
- `ko_effect_rates.csv`
- `per_target_zscored.csv`

Regenerate with `python phase5_liver_fibrosis.py`.
