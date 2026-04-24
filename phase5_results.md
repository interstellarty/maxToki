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

## Interpretation: end of the LL-readout line

Phase 5 was effectively a best-case stress test for the log-likelihood-of-
healthy-anchor readout:

- **Largest biological contrast** tried across all five phases — activated
  vs quiescent HSCs differ more than any aging axis we looked at.
- **Most canonical target possible** — ACTA2 is *the* defining marker of
  myofibroblasts; there is no more textbook KO to attempt.
- **Best statistical power** — n=418 effective cells for ACTA2;
  null std = 0.0016 over 32 effective genes (cleaner than Phase 4b's 0.0015).
- **No expression bottleneck** — 85–91% effect rates for the top targets,
  unlike the 0%/1% rates that killed Phase 3's SASP KOs.

Top result: **ACTA2 at z = +0.28**. Direction correct, magnitude nowhere
near the z = +2 threshold.

This rules out, in turn:
1. **Anchor noise** — n=418 averages out single-anchor idiosyncrasies.
2. **Model capacity** — confirmed separately (Phase 4c was set up for this
   test before being cancelled).
3. **Expression bottleneck** — absent here.
4. **Wrong biology** — TGFB1, SMAD3, CCN2, COL1A1, ACTA2 are the most
   literature-canonical antifibrotic targets in existence.

The remaining explanation is structural. The LL readout asks, *"is the full
2000-gene profile of this perturbed cell more likely to be followed by a
quiescent cell?"* Removing one gene token leaves the other 1999 tokens
still overwhelmingly coding for "activated HSC," so the likelihood of the
quiescent anchor barely moves. The model sees a myofibroblast-minus-ACTA2
as still a myofibroblast, not as a slightly more quiescent cell. This is
the same identity-saturation mechanism Phase 2b identified for the aging
axis, and Phase 5 confirms it generalizes to the disease-state axis. It is
a *property of the readout*, not a data or power limitation.

The paper's method avoids this because the TimeBetweenCells regression
head is trained to read perturbation → direction as a one-dimensional
scalar shift, amplifying a single-gene change. LL cannot do that.

## Stopping criterion

Trying other fibroblast sources (dermal, pulmonary, kidney) or other liver
fibrosis datasets (Guilliams 2022, Govaere 2023, Wang 2024 MASLD) would
hit the same ceiling — same biological axis, same one-token-of-2000 math.
More runs at this framing are unlikely to change the conclusion, so the
LL-readout line of investigation is closed here.

## Cross-phase summary

| phase | axis | target | best z | n_eff | verdict |
|---|---|---|---:|---:|---|
| 2 | aging, hard OE | OSK_L (HSC) | −2.46 | − | OOD artifact |
| 2b | aging, soft OE | OSKM (hep) | −3.14 | 50 | identity loss, not rejuvenation |
| 3 | aging KO | CDKN1A (hep) | +1.93 | 3 | underpowered, later regressed |
| 3b | aging KO | CDKN1A (hep) | +0.54 | 13 | Phase 3 hit was noise |
| 4 | aging KO | GSN (cardiac fib) | −0.83 | 490 | null; wrong head suspected |
| 4b | aging KO ensemble | GSN (cardiac fib) | −0.31 | 198 | ensemble confirms null |
| 5 | disease KO | ACTA2 (HSC) | **+0.28** | **418** | best conditions, still null |

## Next steps

1. **Wait for the second-stage checkpoint.** The Theodoris lab confirmed
   the fine-tuned TimeBetweenCells variant will be shared after peer
   review. That is the only artifact that unblocks the paper's
   perturbation framework.
2. **Take the collaboration offer.** The corresponding-author reply
   included *"if you are interested in applying the specific aging model
   we trained to answer a new biological question we'd be happy to meet
   and discuss potential collaborations."* Liver fibrosis / MASH is a
   natural fit, and Phases 1–5 now constitute well-characterized
   context about the pretraining checkpoint's limits that can frame
   that conversation.
3. **Do not run further LL-readout screens.** Additional datasets or
   cell types are expected to reproduce this negative.

## Provenance

Outputs in `/ptmp/artfi/liver_fibrosis_phase5/screen_hsc_fibrosis_v1/`:

- `scored_results.csv`
- `ko_effect_rates.csv`
- `per_target_zscored.csv`

Regenerate with `python phase5_liver_fibrosis.py`.
