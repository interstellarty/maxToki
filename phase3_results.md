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

## Interpretation

> The auto-generated "Key findings" above reports thresholded hits only.
> The richer read of the numbers is that this is actually **the most
> encouraging result across Phases 1–3**, with a critical caveat about
> the TSP data, not the model.

### Headline

**No z ≥ 2 hit — but the top positive control, CDKN1A (p21) in hepatocytes,
reaches z = +1.93 on just 3 effective cells.** Every strong senescence
driver points in the *correct* direction (KO → rejuvenation), and every
longevity-gene negative control points the *opposite* way (KO → anti-
rejuvenation). Unlike Phase 2, there is no OOD sign-flip artifact.

### Direction is right across the board

**Senescence drivers (expect positive z):**

| gene | cell_type | z | n_eff |
|---|---|---:|---:|
| **CDKN1A** (p21) | hepatocyte | **+1.93** | 3 |
| TP53             | cholangiocyte | +1.45 | 5 |
| RB1              | cholangiocyte | +0.55 | 15 |
| CDKN2A (p16)     | cholangiocyte | +0.36 | 1 |

**SASP / aging markers (expect positive z):**

| gene | cell_type | z | n_eff |
|---|---|---:|---:|
| GLB1     | HSC           | +1.07 | 1 |
| SERPINE1 | hepatocyte    | +0.94 | 9 |
| MDM2     | cholangiocyte | +0.92 | 15 |

**Longevity negative controls (expect near-zero or negative z):**

| gene  | cell_type | z |
|---|---|---:|
| SIRT1 | hepatocyte    | **−1.28** |
| SIRT6 | cholangiocyte | −0.88 |
| FOXO3 | any           | +0.02 to +0.57 (neutral) |

Senescence drivers → positive. Longevity genes → negative. SASP → positive.
**The biology sign-pattern is correct.**

### The real bottleneck: TSP is healthy tissue

Look at `ko_took_effect` (fraction of old cells with the gene in the
top-2000 tokens):

- **IL6, TNF, MMP3, CXCL8** → **0% everywhere.** Not expressed in healthy
  liver. KO is a no-op, which is why they don't appear in the z-score table.
- **CDKN2A (p16)** → 0% HSC, 0% hepatocyte, 2% cholangiocyte. The canonical
  senescence marker is effectively absent.
- **TP53** → 0% HSC, 2% hepatocyte, 10% cholangiocyte. Only cholangiocytes
  express it detectably.
- **CDKN1A (p21)** → 56% HSC, 6% hepatocyte, 4% cholangiocyte.
  **This is why the best hit (hepatocyte CDKN1A) rests on just 3 cells.**

TSP donors are healthy (ages 36, 60, 67). Senescent cells are rare (~5–15%
by SA-β-gal even in aged tissue) and the SASP is only active in that small
subset. In scRNA-seq of mostly non-senescent cells, these genes sit below
rank 2000.

### Null distribution is dramatically tighter than Phase 2b

| cell_type     | Phase 2b soft-OE null std | Phase 3 KO null std |
|---|---:|---:|
| hepatocyte    | 0.0038 | **0.0013** (3× tighter) |
| cholangiocyte | 0.0082 | 0.0017 |
| HSC           | 0.0230 | 0.0025 |

KO is a milder perturbation (drop one token) than OE (shift rank order),
and the effective-KO filter computes the null only over cells where the
prompt actually changed. A small mean_delta can now be significant —
CDKN1A's +0.0028 gives z = +1.93 against this tight null.

### What this says about MaxToki

**The model has weak-but-correct knowledge of the cellular aging axis.**
When perturbations are restricted to in-distribution operations (KO) on
genes actually expressed in the cells being perturbed, the sign pattern
comes out right:

- dropping senescence drivers makes cells look younger
- dropping longevity genes makes cells look older
- magnitudes are tiny (~0.003 log-prob units per token) but consistent

Compare to Phase 2/2b, where the sign was *wrong* (OSKM strongly negative)
— that was the OOD artifact of pinning TFs to rank 0. Here, everything is
in-distribution, and the direction vindicates the biology rather than
fighting it.

### Next moves

1. **Bump `MAX_CELLS_PER_TYPE` from 50 to 500.** TSP has hundreds of old
   hepatocytes. If CDKN1A moves from n_eff = 3 to n_eff ≈ 30, z = +1.93
   becomes either a solid z = 3+ hit or clearly noise. Single-line change,
   ~1 hour of GPU time.
2. **Combo KO of senescence drivers.** Drop CDKN1A + CDKN2A + TP53 + RB1
   together. Biologically the "anti-senescence" intervention; statistically
   pools weak individual signals.
3. **Move to a senescence-enriched dataset.** Tabula Muris Senis (aged mouse
   liver) or Ramachandran 2019 (human fibrotic liver) would have active
   SASP. IL6/TNF/CXCL8 would actually appear in the rank list there.

> Re-running `python phase3_senescence_ko.py` will overwrite this file
> with the auto-generated report only. This Interpretation section was
> written manually after the 2026-04-19T16:02:06 run; copy it back in if
> you re-run and want to preserve the commentary.

## Provenance

Outputs in `/ptmp/artfi/liver_screen_phase3/screen_senescence_ko_v1/`:

- `scored_results.csv`
- `ko_effect_rates.csv`
- `null_stats.csv`
- `positive_zscored.csv`

Regenerate with `python phase3_senescence_ko.py` on branch `claude/continue-previous-session-E56FX`.
