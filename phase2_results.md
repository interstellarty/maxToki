# Phase 2 — Results: TF Overexpression Rejuvenation Screen

**Script:** `phase2_tf_rejuvenation.py`
**Date:** 2026-04-18
**Model:** MaxToki-217M-HF (pretraining-only checkpoint, log-likelihood readout)
**Data:** Tabula Sapiens liver (TSP1_30), 3 cell types, 109 young–old pairs, 71 specs → 7,739 prompts

## Headline

**No Yamanaka or liver-identity TF rejuvenates under this operator.** Strongest
positive z-score is SIRT6 in HSC at z = +0.72. The Yamanaka combos go the
*wrong way* — they are the most significantly negative specs in the entire
screen.

| spec | cell_type | mean_delta | z |
|---|---|---:|---:|
| OE_COMBO:OSK_L   | hepatic stellate cell      | −0.0085 | **−2.46** |
| OE_COMBO:OSKM    | hepatic stellate cell      | −0.0065 | **−2.00** |
| OE:SOX2          | hepatic stellate cell      | −0.0059 | −1.84 |
| OE_COMBO:OSK     | hepatic stellate cell      | −0.0048 | −1.56 |
| OE_COMBO:OSK_L   | hepatocyte                 | −0.0126 | −1.03 |
| OE_COMBO:OSKM    | hepatocyte                 | −0.0078 | −0.54 |
| —                | —                          | —       | — |
| OE:SIRT6         | hepatic stellate cell      | +0.0046 | **+0.72** (top positive) |
| OE:SIRT1         | hepatocyte                 | +0.0036 | +0.65 |
| OE:SIRT6         | hepatocyte                 | +0.0023 | +0.51 |
| OE:HNF1A         | intrahepatic cholangiocyte | +0.0034 | +0.49 |

Null distribution (50 random-gene OEs) per cell type:

| cell_type | null_mean | null_std |
|---|---:|---:|
| hepatic stellate cell      | +0.0016 | 0.0041 |
| hepatocyte                 | −0.0026 | 0.0096 |
| intrahepatic cholangiocyte | +0.00001 | 0.0070 |

## Interpretation

### Why this is informative, not broken

The `make_overexpression_spec` operator **promotes the gene to rank 0** — i.e.
makes it the single most-expressed gene in the cell. In real rank-value-encoded
scRNA-seq, rank 0 is dominated by housekeeping genes or cell-type-specific
markers (albumin in hepatocytes, collagen in HSCs, keratins in cholangiocytes);
transcription factors normally sit **mid-to-low rank** because their absolute
mRNA counts are low. So pushing a TF to rank 0 creates an out-of-distribution
profile, and pushing *four* TFs to rank 0–3 (OSKM) pushes further OOD. The
fact that the most negative z-scores scale with combo size — **1 gene (SOX2)
< 3 genes (OSK) < 4 genes (OSKM / OSK_L)** — is the signature of an OOD
artifact, not a biological signal about reprogramming.

The model is saying **"this transcriptional profile is unphysiological"**, not
"TFs don't rejuvenate".

### Null outlier inflates hepatocyte variance

One random-gene pick, **ENSG00000233115**, has `mean_delta = −0.036` in
cholangiocyte and **−0.059** in hepatocyte (std ≈ 0.09). This looks like a
long-noncoding RNA or pseudogene whose promotion to rank 0 is catastrophic for
the model. It inflates the hepatocyte null std from ~0.004 to 0.0096, which
deflates every hepatocyte z-score by ~2.4×. Worth re-running the analysis
with winsorization or outlier trimming of the null set.

### HNF4A (hepatocyte master regulator) is neutral-to-negative in hepatocytes

`OE:HNF4A` in hepatocytes: `mean_delta = −0.0048`, z = −0.23.
HNF4A is the canonical hepatocyte identity TF and should produce the
*clearest* positive signal if the operator modeled biological overexpression.
It doesn't. This directly confirms that the operator is not capturing
biological OE semantics for TFs.

## What MaxToki is (and isn't) modeling here

- **Is**: an autoregressive LM over rank-value-encoded gene tokens. It learns
  the joint distribution of gene rank orderings observed in single-cell data.
- **Isn't**: a causal/regulatory model. It has no notion of "gene X activates
  gene Y". Overexpressing a TF in the token sequence doesn't trigger
  downstream transcriptional changes — it just rearranges rank order.

So a successful "OE → rejuvenation" signal would require the training data to
contain many real cells in which the target TF happens to be at rank 0 **and**
those cells also happen to look young. TFs at rank 0 are rare in training →
OE-to-rank-0 creates OOD prompts → log-prob drops.

## Suggested next experiments

1. **Softer OE operator.** Instead of rank-0 promotion, define
   `make_soft_overexpression_spec(gene, boost_ranks=K)` that moves the gene up
   by K positions (e.g. K=50 or rank ×0.5). Physiologically realistic for TFs
   and keeps prompts in-distribution.
2. **Signature-level OE.** For each TF, overexpress its known target set
   (from TRRUST / DoRothEA) rather than the TF itself. This tests whether the
   downstream transcriptional consequence of TF activation looks younger.
3. **Winsorize the null.** Redo z-scoring after dropping the top/bottom 5% of
   the null distribution; re-report the top hits. Removes single-gene outliers
   like ENSG00000233115.
4. **Reverse screen.** KO of *aging-associated* genes (CDKN2A, CDKN1A, GLB1,
   LMNB1) rather than OE of rejuvenation factors — KO is in-distribution
   (it just drops a token) and should give cleaner signal.

## Provenance

Outputs written to `/ptmp/$USER/liver_screen_phase2/screen_rejuv_tfs_v1/`:

- `scored_results.csv` — 7,739 rows, one per prompt
- `summary_per_spec.csv` — per-(spec, cell_type) mean/std/n
- `null_stats.csv` — per-cell-type null mean/std/count
- `positive_tfs_zscored.csv` — final z-score table

Reproduce with:
```bash
python phase2_tf_rejuvenation.py
```
on the branch `claude/continue-previous-session-E56FX`.
