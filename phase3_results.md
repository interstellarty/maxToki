# Phase 3 — Senescence-Gene Knockout Screen: Results

**Run:** 2026-04-19T16:02:06  
**Script:** `phase3_senescence_ko.py`  
**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  
**Data:** TSP1_30 liver — 109 young–old pairs across 3 cell types  
**Specs:** 1 baseline + 12 senescence/SASP/aging KOs + 3 longevity-negative-control KOs + 50 null KOs = 66 total  
**Prompts scored:** 7194

## Pairs per cell type

| cell_type | n_old_cells |
| --- | --- |
| intrahepatic cholangiocyte | 50 |
| hepatocyte | 50 |
| hepatic stellate cell | 9 |

## KO effect rates (fraction of old cells where the KO dropped a token)

A gene with low effect rate is absent from most old cells' top-2000 tokens — any z-score based on it rests on few cells.

| gene_symbol | gene_category | hepatic stellate cell | hepatocyte | intrahepatic cholangiocyte |
| --- | --- | --- | --- | --- |
| SIRT6 | longevity_neg_ctrl | 0.00 | 0.02 | 0.04 |
| SIRT1 | longevity_neg_ctrl | 0.22 | 0.08 | 0.22 |
| SERPINE1 | sasp | 0.11 | 0.18 | 0.08 |
| CCL2 | sasp | 0.11 | 0.00 | 0.36 |
| FOXO3 | longevity_neg_ctrl | 0.56 | 0.44 | 0.48 |
| CDKN1A | senescence_driver | 0.56 | 0.06 | 0.04 |
| MDM2 | aging_marker | 0.22 | 0.48 | 0.30 |
| IL6 | sasp | 0.00 | 0.00 | 0.00 |
| RB1 | senescence_driver | 0.22 | 0.10 | 0.30 |
| TP53 | senescence_driver | 0.00 | 0.02 | 0.10 |
| CDKN2A | senescence_driver | 0.00 | 0.00 | 0.02 |
| MMP3 | sasp | 0.00 | 0.00 | 0.00 |
| CXCL8 | sasp | 0.00 | 0.00 | 0.18 |
| GLB1 | aging_marker | 0.11 | 0.22 | 0.22 |
| TNF | sasp | 0.00 | 0.00 | 0.00 |

## Null distribution (50 random-gene KOs, effective KOs only)

| cell_type | null_mean | null_std | n_effective_genes |
| --- | --- | --- | --- |
| hepatic stellate cell | -0.0005 | 0.0025 | 24 |
| hepatocyte | 0.0004 | 0.0013 | 28 |
| intrahepatic cholangiocyte | 0.0004 | 0.0017 | 35 |

## Z-scored positives and negative controls

`delta_vs_baseline > 0` = perturbed old cell looks more like the young anchor (rejuvenation signal). `|z| > 2` is notable.

| gene_symbol | gene_category | cell_type | spec_class | mean_delta | n_effective | null_mean | null_std | z_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CDKN1A | senescence_driver | hepatocyte | positive | 0.0028 | 3 | 0.0004 | 0.0013 | 1.9282 |
| TP53 | senescence_driver | intrahepatic cholangiocyte | positive | 0.0029 | 5 | 0.0004 | 0.0017 | 1.4508 |
| GLB1 | aging_marker | hepatic stellate cell | positive | 0.0022 | 1 | -0.0005 | 0.0025 | 1.0659 |
| SERPINE1 | sasp | hepatocyte | positive | 0.0016 | 9 | 0.0004 | 0.0013 | 0.9363 |
| MDM2 | aging_marker | intrahepatic cholangiocyte | positive | 0.0020 | 15 | 0.0004 | 0.0017 | 0.9212 |
| FOXO3 | longevity_neg_ctrl | intrahepatic cholangiocyte | neg_control | 0.0014 | 24 | 0.0004 | 0.0017 | 0.5720 |
| RB1 | senescence_driver | intrahepatic cholangiocyte | positive | 0.0014 | 15 | 0.0004 | 0.0017 | 0.5482 |
| CDKN1A | senescence_driver | hepatic stellate cell | positive | 0.0005 | 5 | -0.0005 | 0.0025 | 0.4118 |
| CDKN2A | senescence_driver | intrahepatic cholangiocyte | positive | 0.0011 | 1 | 0.0004 | 0.0017 | 0.3621 |
| MDM2 | aging_marker | hepatic stellate cell | positive | 0.0002 | 2 | -0.0005 | 0.0025 | 0.2686 |
| CCL2 | sasp | hepatic stellate cell | positive | 0.0001 | 1 | -0.0005 | 0.0025 | 0.2621 |
| SERPINE1 | sasp | hepatic stellate cell | positive | 0.0001 | 1 | -0.0005 | 0.0025 | 0.2586 |
| FOXO3 | longevity_neg_ctrl | hepatic stellate cell | neg_control | 0.0001 | 5 | -0.0005 | 0.0025 | 0.2354 |
| RB1 | senescence_driver | hepatocyte | positive | 0.0006 | 5 | 0.0004 | 0.0013 | 0.1385 |
| SIRT6 | longevity_neg_ctrl | hepatocyte | neg_control | 0.0005 | 1 | 0.0004 | 0.0013 | 0.1131 |
| FOXO3 | longevity_neg_ctrl | hepatocyte | neg_control | 0.0004 | 22 | 0.0004 | 0.0013 | 0.0253 |
| SIRT1 | longevity_neg_ctrl | hepatic stellate cell | neg_control | -0.0005 | 2 | -0.0005 | 0.0025 | 0.0077 |
| GLB1 | aging_marker | intrahepatic cholangiocyte | positive | 0.0002 | 11 | 0.0004 | 0.0017 | -0.1432 |
| SERPINE1 | sasp | intrahepatic cholangiocyte | positive | 0.0001 | 4 | 0.0004 | 0.0017 | -0.1947 |
| MDM2 | aging_marker | hepatocyte | positive | -0.0002 | 24 | 0.0004 | 0.0013 | -0.4208 |
| CXCL8 | sasp | intrahepatic cholangiocyte | positive | -0.0003 | 9 | 0.0004 | 0.0017 | -0.4584 |
| SIRT1 | longevity_neg_ctrl | intrahepatic cholangiocyte | neg_control | -0.0004 | 11 | 0.0004 | 0.0017 | -0.5135 |
| GLB1 | aging_marker | hepatocyte | positive | -0.0004 | 11 | 0.0004 | 0.0013 | -0.6152 |
| CCL2 | sasp | intrahepatic cholangiocyte | positive | -0.0008 | 18 | 0.0004 | 0.0017 | -0.7125 |
| SIRT6 | longevity_neg_ctrl | intrahepatic cholangiocyte | neg_control | -0.0011 | 2 | 0.0004 | 0.0017 | -0.8777 |
| CDKN1A | senescence_driver | intrahepatic cholangiocyte | positive | -0.0011 | 2 | 0.0004 | 0.0017 | -0.9173 |
| RB1 | senescence_driver | hepatic stellate cell | positive | -0.0034 | 2 | -0.0005 | 0.0025 | -1.1347 |
| SIRT1 | longevity_neg_ctrl | hepatocyte | neg_control | -0.0013 | 4 | 0.0004 | 0.0013 | -1.2832 |
| TP53 | senescence_driver | hepatocyte | positive | -0.0015 | 1 | 0.0004 | 0.0013 | -1.4421 |

## Key findings

No positive-control KO passed |z| ≥ 2.0 in any cell type. MaxToki does not distinguish young vs. old liver cells along the canonical senescence axis at 217M scale.

No longevity-gene KO passed |z| ≥ 2.0. Negative controls behave as expected.

## Provenance

Outputs in `/ptmp/artfi/liver_screen_phase3/screen_senescence_ko_v1/`:

- `scored_results.csv`
- `ko_effect_rates.csv`
- `null_stats.csv`
- `positive_zscored.csv`

Regenerate with `python phase3_senescence_ko.py` on branch `claude/continue-previous-session-E56FX`.
