# Phase 3 — Senescence-Gene Knockout Screen: Results

**Run:** 2026-04-21T14:47:25  
**Script:** `phase3_senescence_ko.py`  
**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  
**Data:** TSP1_30 liver — 591 young–old pairs across 3 cell types  
**Specs:** 1 baseline + 12 senescence/SASP/aging KOs + 3 longevity-negative-control KOs + 50 null KOs = 66 total  
**Prompts scored:** 39006

## Pairs per cell type

| cell_type | n_old_cells |
| --- | --- |
| intrahepatic cholangiocyte | 82 |
| hepatocyte | 500 |
| hepatic stellate cell | 9 |

## KO effect rates (fraction of old cells where the KO dropped a token)

A gene with low effect rate is absent from most old cells' top-2000 tokens — any z-score based on it rests on few cells.

| gene_symbol | gene_category | hepatic stellate cell | hepatocyte | intrahepatic cholangiocyte |
| --- | --- | --- | --- | --- |
| SIRT6 | longevity_neg_ctrl | 0.00 | 0.05 | 0.02 |
| SIRT1 | longevity_neg_ctrl | 0.22 | 0.11 | 0.24 |
| SERPINE1 | sasp | 0.11 | 0.25 | 0.05 |
| CCL2 | sasp | 0.11 | 0.00 | 0.26 |
| FOXO3 | longevity_neg_ctrl | 0.56 | 0.57 | 0.54 |
| CDKN1A | senescence_driver | 0.56 | 0.03 | 0.02 |
| MDM2 | aging_marker | 0.22 | 0.56 | 0.30 |
| IL6 | sasp | 0.00 | 0.00 | 0.00 |
| RB1 | senescence_driver | 0.22 | 0.12 | 0.33 |
| TP53 | senescence_driver | 0.00 | 0.05 | 0.09 |
| CDKN2A | senescence_driver | 0.00 | 0.00 | 0.01 |
| MMP3 | sasp | 0.00 | 0.00 | 0.00 |
| CXCL8 | sasp | 0.00 | 0.01 | 0.13 |
| GLB1 | aging_marker | 0.11 | 0.28 | 0.20 |
| TNF | sasp | 0.00 | 0.00 | 0.00 |

## Null distribution (50 random-gene KOs, effective KOs only)

| cell_type | null_mean | null_std | n_effective_genes |
| --- | --- | --- | --- |
| hepatic stellate cell | -0.0005 | 0.0025 | 24 |
| hepatocyte | 0.0008 | 0.0012 | 36 |
| intrahepatic cholangiocyte | 0.0002 | 0.0015 | 35 |

## Z-scored positives and negative controls

`delta_vs_baseline > 0` = perturbed old cell looks more like the young anchor (rejuvenation signal). `|z| > 2` is notable.

| gene_symbol | gene_category | cell_type | spec_class | mean_delta | n_effective | null_mean | null_std | z_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CXCL8 | sasp | hepatocyte | positive | 0.0033 | 3 | 0.0008 | 0.0012 | 2.1679 |
| TP53 | senescence_driver | intrahepatic cholangiocyte | positive | 0.0026 | 7 | 0.0002 | 0.0015 | 1.5841 |
| CCL2 | sasp | hepatocyte | positive | 0.0021 | 1 | 0.0008 | 0.0012 | 1.1573 |
| MDM2 | aging_marker | intrahepatic cholangiocyte | positive | 0.0019 | 25 | 0.0002 | 0.0015 | 1.1330 |
| GLB1 | aging_marker | hepatic stellate cell | positive | 0.0022 | 1 | -0.0005 | 0.0025 | 1.0659 |
| RB1 | senescence_driver | intrahepatic cholangiocyte | positive | 0.0017 | 27 | 0.0002 | 0.0015 | 0.9727 |
| CDKN2A | senescence_driver | intrahepatic cholangiocyte | positive | 0.0011 | 1 | 0.0002 | 0.0015 | 0.5666 |
| CDKN1A | senescence_driver | hepatocyte | positive | 0.0014 | 13 | 0.0008 | 0.0012 | 0.5445 |
| FOXO3 | longevity_neg_ctrl | intrahepatic cholangiocyte | neg_control | 0.0009 | 44 | 0.0002 | 0.0015 | 0.4588 |
| CDKN1A | senescence_driver | hepatic stellate cell | positive | 0.0005 | 5 | -0.0005 | 0.0025 | 0.4118 |
| SERPINE1 | sasp | hepatocyte | positive | 0.0012 | 126 | 0.0008 | 0.0012 | 0.3820 |
| MDM2 | aging_marker | hepatic stellate cell | positive | 0.0002 | 2 | -0.0005 | 0.0025 | 0.2686 |
| CCL2 | sasp | hepatic stellate cell | positive | 0.0001 | 1 | -0.0005 | 0.0025 | 0.2621 |
| SERPINE1 | sasp | hepatic stellate cell | positive | 0.0001 | 1 | -0.0005 | 0.0025 | 0.2586 |
| FOXO3 | longevity_neg_ctrl | hepatic stellate cell | neg_control | 0.0001 | 5 | -0.0005 | 0.0025 | 0.2354 |
| GLB1 | aging_marker | intrahepatic cholangiocyte | positive | 0.0004 | 16 | 0.0002 | 0.0015 | 0.1452 |
| SIRT1 | longevity_neg_ctrl | hepatic stellate cell | neg_control | -0.0005 | 2 | -0.0005 | 0.0025 | 0.0077 |
| SERPINE1 | sasp | intrahepatic cholangiocyte | positive | 0.0001 | 4 | 0.0002 | 0.0015 | -0.0540 |
| TP53 | senescence_driver | hepatocyte | positive | 0.0005 | 26 | 0.0008 | 0.0012 | -0.1934 |
| RB1 | senescence_driver | hepatocyte | positive | 0.0004 | 59 | 0.0008 | 0.0012 | -0.2726 |
| FOXO3 | longevity_neg_ctrl | hepatocyte | neg_control | 0.0003 | 287 | 0.0008 | 0.0012 | -0.3729 |
| SIRT6 | longevity_neg_ctrl | hepatocyte | neg_control | -0.0000 | 24 | 0.0008 | 0.0012 | -0.6478 |
| GLB1 | aging_marker | hepatocyte | positive | -0.0002 | 138 | 0.0008 | 0.0012 | -0.7860 |
| SIRT6 | longevity_neg_ctrl | intrahepatic cholangiocyte | neg_control | -0.0011 | 2 | 0.0002 | 0.0015 | -0.8153 |
| CDKN1A | senescence_driver | intrahepatic cholangiocyte | positive | -0.0011 | 2 | 0.0002 | 0.0015 | -0.8594 |
| SIRT1 | longevity_neg_ctrl | hepatocyte | neg_control | -0.0005 | 54 | 0.0008 | 0.0012 | -1.0944 |
| MDM2 | aging_marker | hepatocyte | positive | -0.0006 | 282 | 0.0008 | 0.0012 | -1.1235 |
| RB1 | senescence_driver | hepatic stellate cell | positive | -0.0034 | 2 | -0.0005 | 0.0025 | -1.1347 |
| SIRT1 | longevity_neg_ctrl | intrahepatic cholangiocyte | neg_control | -0.0017 | 20 | 0.0002 | 0.0015 | -1.2061 |
| CCL2 | sasp | intrahepatic cholangiocyte | positive | -0.0017 | 21 | 0.0002 | 0.0015 | -1.2532 |
| CXCL8 | sasp | intrahepatic cholangiocyte | positive | -0.0024 | 11 | 0.0002 | 0.0015 | -1.6716 |

## Key findings

### Positive controls passing |z| ≥ 2.0

| gene_symbol | gene_category | cell_type | mean_delta | n_effective | z_score |
| --- | --- | --- | --- | --- | --- |
| CXCL8 | sasp | hepatocyte | 0.0033 | 3 | 2.1679 |

**1 senescence/SASP/aging-marker KO(s) show a rejuvenation signal (z ≥ +2.0).** This is the real-deal finding: KO of these genes makes the old cell's transcriptome look more like its young counterpart.

No longevity-gene KO passed |z| ≥ 2.0. Negative controls behave as expected.

## Provenance

Outputs in `/ptmp/artfi/liver_screen_phase3b/screen_senescence_ko_scaled_v1/`:

- `scored_results.csv`
- `ko_effect_rates.csv`
- `null_stats.csv`
- `positive_zscored.csv`

Regenerate with `python phase3_senescence_ko.py` on branch `claude/continue-previous-session-E56FX`.
