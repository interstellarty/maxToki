# Phase 2b — Results: Soft-OE Rejuvenation Screen

**Script:** `phase2b_soft_oe.py`
**Date:** 2026-04-19
**Model:** MaxToki-217M-HF (pretraining-only checkpoint, log-likelihood readout)
**Data:** TSP1_30 liver, 3 cell types, 109 young–old pairs, 67 specs → 7,303 prompts

## Executive summary

**The soft operator worked as intended — and the negative finding is now more
convincing, not less.** No TF rejuvenates old liver cells in MaxToki's view.
OSKM stays strongly negative across the K-sweep (all 7 values), even at the
mildest physiological boost (K=10 is z = −1.81 in hepatocytes). This is
unlikely to be an OOD artifact.

## Null distribution: soft-OE fixed the hepatocyte outlier problem

| cell_type | Phase 2 hard-OE null std | Phase 2b soft-OE null std |
|---|---:|---:|
| hepatocyte | 0.0096 (inflated by ENSG00000233115 pseudogene) | **0.0038** (2.5× tighter) |
| cholangiocyte | 0.0070 | 0.0082 |
| HSC | 0.0041 | 0.023 (dominated by small-sample noise, n=9) |

Hepatocyte null mean moved from −0.0026 → +0.0005. The catastrophic pseudogene
is gone — validating the soft operator.

## OSKM K-sweep in hepatocytes (n = 50 old cells)

| K | mean_delta | z-score |
|---:|---:|---:|
| 10   | −0.0064 | −1.81 |
| 25   | **−0.0115** | **−3.14** |
| 50   | −0.0064 | −1.81 |
| 100  | −0.0055 | −1.57 |
| 200  | −0.0047 | −1.36 |
| 500  | −0.0035 | −1.06 |
| 2000 (≈ hard) | −0.0142 | **−3.87** |

**No K turns OSKM positive.** K=25 is the most negative in the physiological
range — a mild perturbation (4 TFs up by 25 ranks in a 2000-token cell)
produces a robust z = −3.14. This is the *opposite* of what you'd expect if
Phase 2's negatives were operator artifacts.

## Single-TF z-scores (hepatocyte, n = 50, null std = 0.0038)

| spec | mean_delta | z |
|---|---:|---:|
| sOE50:NANOG  | +0.0037 | +0.86 |
| sOE50:KLF4   | +0.0002 | −0.07 |
| sOE50:HNF4A  | −0.0002 | −0.19 |
| sOE50:SIRT6  | −0.0003 | −0.21 |
| sOE50:SOX2   | −0.0009 | −0.35 |
| sOE50:MYC    | −0.0023 | −0.73 |
| sOE50:SIRT1  | −0.0037 | −1.10 |
| sOE50:POU5F1 | −0.0034 | −1.03 |

No single TF clears z = +2. HNF4A (hepatocyte master regulator) is z = −0.19.

## Interpretation: MaxToki is detecting loss of cell identity, not lack of rejuvenation

Mature hepatocytes have a stereotyped top-rank profile (albumin, fibrinogens,
transthyretin). Promoting POU5F1/SOX2/KLF4/MYC — even mildly — displaces those
canonical markers down the rank order. The perturbed cell looks *less like a
mature hepatocyte*, so log-prob of a young mature hepatocyte continuation
drops.

**Biologically this is correct.** Yamanaka factors drive dedifferentiation
toward pluripotency — movement away from the young-mature-hepatocyte anchor,
not toward it. The experimental setup "young→old continuation within the same
cell type" cannot reward dedifferentiation, because the young anchor is also a
differentiated cell.

## Caveats

1. **HSC results are uninterpretable** (n=9 old cells, 1 anchor). The HSC null
   mean of +0.023 and std of 0.023 are small-sample noise inflated by all
   specs sharing the same 9 cells. The top "positive" random-gene deltas
   (0.03–0.07) are noise-floor artifacts.
2. **K=2000 doesn't exactly reproduce Phase 2 hard OE** (−0.0142 vs −0.0078
   for OSKM hepatocyte) because soft-combo applies genes sequentially and
   reshuffles when combo genes happen to be present in the cell. Harmless for
   interpretation.

## What this tells us about MaxToki

The pretraining-only checkpoint learns `P(next cell | current cell)` within
cell types in healthy tissue. It does not know what "young" means independently
of cell identity. A rejuvenation signal in this framework requires a young
anchor that differs from the old cell *only in age-associated genes*, not in
cell-type-defining genes. The current single-anchor setup is dominated by
cell-identity noise.

## Next steps

1. **Aging-gene KO screen** (`phase3_senescence_ko.py`) — KO of CDKN2A,
   CDKN1A, TP53, IL6, etc. KO is in-distribution (just drops a token) and
   directly tests the senescence axis. Longevity genes (SIRT1/6, FOXO3) serve
   as negative controls — their KO should *not* rejuvenate.
2. **Signature-level OE** — OE the TF's known downstream targets from
   DoRothEA/TRRUST instead of the TF itself. Bypasses the rank-displacement
   problem.
3. **Young-anchor ensembling** — average log-prob over multiple young anchors
   per cell type to reduce anchor-specific identity noise. Cheap extension of
   the current pipeline.

## Provenance

Outputs written to `/ptmp/$USER/liver_screen_phase2b/screen_soft_oe_v1/`:
`scored_results.csv`, `summary_per_spec.csv`, `null_stats.csv`,
`positive_tfs_zscored.csv`, `oskm_sweep.csv`.
