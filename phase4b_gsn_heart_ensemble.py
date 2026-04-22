# %% [markdown]
# # Phase 4b: GSN cardiac-fibroblast replication — multi-anchor ensemble + soft-OE
#
# Phase 4 produced a clean null (z = −0.83, effectively zero) with a
# well-powered single-anchor setup. Two candidate reasons the paper's signal
# didn't surface:
#   1. One young anchor carries cell-identity noise that may drown a subtle age
#      signal.
#   2. Paper may have tested OE, not KO.
#
# This phase changes only those two variables and keeps everything else
# identical (same h5ad, same age split, same cardiac-fibroblast filter, same
# pretraining-only LL readout).
#
# Changes vs Phase 4:
#   - N_ANCHORS = 10 young anchors per old cell (vs. 1). delta_vs_baseline is
#     averaged across anchors for each old cell before z-scoring.
#   - Add `sOE50:GSN` spec (soft-OE by 50 ranks, same operator as Phase 2b).
#   - Add matched soft-OE null: apply sOE50 to the same 50 random genes used
#     for the KO null, so the two arms are directly comparable.
#   - MAX_CELLS reduced 500 → 200 to keep runtime bounded (200 × 10 × 103 ≈
#     206k prompts, ~2h on A100).
#
# Reuses the tokenized datasets from Phase 4 at
# `/ptmp/$USER/heart_screen_phase4/tokenized_{young,old}` so no retokenization.
#
# Outputs:
#   - `/ptmp/$USER/heart_screen_phase4b/` (CSVs + screen dataset)
#   - `phase4b_results.md` at the repo root

# %% — Imports and config
import json
import random
import re
import warnings
from datetime import datetime
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

warnings.filterwarnings("ignore", category=FutureWarning)

RESOURCES = Path("/workspace/resources")
TOKEN_DICT_PATH = RESOURCES / "token_dictionary_v1.json"
ENSEMBL_MAP_PATH = RESOURCES / "ensembl_mapping_dict_v1.json"
HF_MODEL_PATH = Path("/ptmp/artfi/models/maxtoki-hf/MaxToki-217M-HF")

PHASE4_DIR = Path("/ptmp/artfi/heart_screen_phase4")
OUTPUT_DIR = Path("/ptmp/artfi/heart_screen_phase4b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_MAX_TOKENS = 2000
MAX_CELLS = 200
N_ANCHORS = 10
N_NULL_GENES = 50
SOFT_OE_BOOST = 50
SEED = 42
Z_NOTABLE = 2.0
TARGET_GENE = "GSN"

# %% — Load tokenized datasets from Phase 4
ds_young_dir = next((PHASE4_DIR / "tokenized_young").glob("*.dataset"))
ds_old_dir = next((PHASE4_DIR / "tokenized_old").glob("*.dataset"))
ds_young = datasets.load_from_disk(str(ds_young_dir))
ds_old = datasets.load_from_disk(str(ds_old_dir))
print(f"Loaded: {len(ds_young)} young, {len(ds_old)} old (from Phase 4 tokenization)")

# %% — Tokenizer and perturbation helpers
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
from bionemo.maxtoki.perturb import (
    CellPair, build_screen_dataset_scoring, score_screen,
    make_noop_spec, make_knockout_spec, make_soft_overexpression_spec,
)

with open(TOKEN_DICT_PATH) as f:
    token_dict = json.load(f)
with open(ENSEMBL_MAP_PATH) as f:
    ensembl_map = json.load(f)
tokenizer = MaxTokiTokenizer(token_dict)


def symbol_to_ensembl(symbol):
    eid = ensembl_map.get(symbol.upper())
    if eid is None:
        raise KeyError(f"{symbol!r} not in ensembl_mapping_dict_v1.json")
    return eid


def _truncate(tokens):
    if len(tokens) <= CELL_MAX_TOKENS:
        return list(tokens)
    bos = [tokens[0]] if tokens and tokens[0] == tokenizer.bos_id else []
    eos = [tokens[-1]] if tokens and tokens[-1] == tokenizer.eos_id else []
    body = list(tokens[len(bos): len(tokens) - len(eos)])
    body = body[: CELL_MAX_TOKENS - len(bos) - len(eos)]
    return [*bos, *body, *eos]


# %% — Pick N_ANCHORS young cells spread across the length distribution
young_lengths = np.array([len(r["input_ids"]) for r in ds_young])
sorted_idx = np.argsort(young_lengths)
# N_ANCHORS evenly spaced quantiles across the length distribution to span the
# diversity of young fibroblasts rather than cluster near the median.
quantile_positions = np.linspace(0, len(sorted_idx) - 1, N_ANCHORS).astype(int)
anchor_idxs = sorted_idx[quantile_positions]
anchors = [_truncate(ds_young[int(i)]["input_ids"]) for i in anchor_idxs]
print(f"Picked {len(anchors)} anchors at idxs {anchor_idxs.tolist()}; "
      f"lengths {[len(a) for a in anchors]}")

# %% — Build pairs: N_ANCHORS × n_old
n_old = min(len(ds_old), MAX_CELLS)
pairs = []
for anchor_j, anchor_tokens in enumerate(anchors):
    for i in range(n_old):
        pairs.append(CellPair(
            young_tokens=anchor_tokens,
            old_tokens=_truncate(ds_old[i]["input_ids"]),
            cell_id=f"fib_old{i}_anc{anchor_j}",
            metadata={"cell_type": "cardiac_fibroblast", "old_idx": i, "anchor_idx": anchor_j},
        ))
print(f"Built {len(pairs)} pairs ({N_ANCHORS} anchors × {n_old} old cells).")

# %% — Specs: baseline + KO:GSN + sOE50:GSN + 50 null KO + 50 null sOE
specs = [make_noop_spec()]

gsn_eid = symbol_to_ensembl(TARGET_GENE)
if gsn_eid not in token_dict:
    raise SystemExit(f"{TARGET_GENE} ({gsn_eid}) not in MaxToki vocab")

specs.append(make_knockout_spec(tokenizer, gsn_eid))
specs.append(make_soft_overexpression_spec(tokenizer, gsn_eid, boost_ranks=SOFT_OE_BOOST))
print(f"Target specs: KO:{gsn_eid}, sOE{SOFT_OE_BOOST}:{gsn_eid} (GSN)")

_rng = random.Random(SEED)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_null_pool = [e for e in _vocab_ensgs if e != gsn_eid]
null_ensgs = _rng.sample(_null_pool, N_NULL_GENES)

# 50 null KOs
for eid in null_ensgs:
    specs.append(make_knockout_spec(tokenizer, eid))

# 50 null sOE — apply sOE50 to the *same* 50 random Ensembl IDs so operator is
# the only thing that differs between the KO null and the sOE null.
for eid in null_ensgs:
    specs.append(make_soft_overexpression_spec(tokenizer, eid, boost_ranks=SOFT_OE_BOOST))

print(f"Total specs: {len(specs)} "
      f"(1 baseline + 2 target + {N_NULL_GENES} null-KO + up-to-{N_NULL_GENES} null-sOE)")

# %% — Build screen dataset and score
SCREEN_DIR = OUTPUT_DIR / "screen_gsn_ensemble_v1"
ds_path, manifest_path = build_screen_dataset_scoring(
    tokenizer, pairs, specs, output_dir=SCREEN_DIR, model_input_size=4096,
)
manifest = pd.read_csv(manifest_path)
print(f"Screen: {len(manifest)} prompts -> {SCREEN_DIR}")

print("Loading MaxToki-217M-HF...")
model = AutoModelForCausalLM.from_pretrained(
    str(HF_MODEL_PATH), torch_dtype=torch.bfloat16,
).to("cuda").eval()
print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

print("Scoring...")
df = score_screen(model, ds_path, manifest_path, device="cuda")
df.to_csv(SCREEN_DIR / "scored_results.csv", index=False)

# %% — Parse old_idx/anchor_idx from cell_id, compute ko_took_effect
df["old_idx"] = df["cell_id"].str.extract(r"old(\d+)").astype(int)
df["anchor_idx"] = df["cell_id"].str.extract(r"anc(\d+)").astype(int)

# ko_took_effect is per-(spec, old_idx, anchor_idx). For KO it's invariant
# across anchors (depends only on the old cell's tokens); for sOE the same
# holds.  We compare prompt_length against the baseline for the same
# (old_idx, anchor_idx).
baseline_lengths = (
    df[df["spec_name"] == "baseline"]
    .set_index(["old_idx", "anchor_idx"])["prompt_length"]
    .to_dict()
)
df["_key"] = list(zip(df["old_idx"], df["anchor_idx"]))
df["baseline_len"] = df["_key"].map(baseline_lengths)
df["ko_took_effect"] = df["prompt_length"] != df["baseline_len"]
df = df.drop(columns=["_key", "baseline_len"])

# %% — Ensemble across anchors: average delta_vs_baseline per (spec, old_idx)
# Only average over anchors where the perturbation took effect. If an old cell
# has zero anchors with effect for a given spec, drop it.
eff = df[df["ko_took_effect"]]
per_old = (
    eff.groupby(["spec_name", "old_idx"])["delta_vs_baseline"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "delta_ens", "count": "n_anchors_eff"})
)
# Drop rows where < 3 anchors contributed (noise-dominated averages).
per_old = per_old[per_old["n_anchors_eff"] >= 3]
print(f"After ensembling: {len(per_old)} (spec, old_idx) cells with ≥3 effective anchors")

# Tag spec classes
gsn_ko = f"KO:{gsn_eid}"
gsn_soe = f"sOE{SOFT_OE_BOOST}:{gsn_eid}"
null_ko_specs = {f"KO:{e}" for e in null_ensgs}
null_soe_specs = {f"sOE{SOFT_OE_BOOST}:{e}" for e in null_ensgs}

def classify(spec_name):
    if spec_name == "baseline":
        return "baseline"
    if spec_name == gsn_ko:
        return "target_ko"
    if spec_name == gsn_soe:
        return "target_soe"
    if spec_name in null_ko_specs:
        return "null_ko"
    if spec_name in null_soe_specs:
        return "null_soe"
    return "other"


per_old["spec_class"] = per_old["spec_name"].map(classify)
per_old.to_csv(SCREEN_DIR / "per_old_ensemble.csv", index=False)

# %% — Z-score GSN KO vs null-KO, and GSN sOE vs null-sOE
def zscore_arm(target_class, null_class):
    """Null: per-gene mean of ensembled delta_ens, then mean/std across genes."""
    null_df = per_old[per_old["spec_class"] == null_class]
    if null_df.empty:
        return dict(mean=float("nan"), std=float("nan"), n_cells=0, null_mean=float("nan"),
                    null_std=float("nan"), n_null_genes=0, z=float("nan"))
    null_per_gene = null_df.groupby("spec_name")["delta_ens"].mean()
    null_mean, null_std = null_per_gene.mean(), null_per_gene.std()

    tgt_df = per_old[per_old["spec_class"] == target_class]
    if tgt_df.empty:
        return dict(mean=float("nan"), std=float("nan"), n_cells=0,
                    null_mean=null_mean, null_std=null_std,
                    n_null_genes=len(null_per_gene), z=float("nan"))
    tgt_mean = tgt_df["delta_ens"].mean()
    tgt_std = tgt_df["delta_ens"].std()
    z = (tgt_mean - null_mean) / null_std if null_std > 0 else float("nan")
    return dict(mean=tgt_mean, std=tgt_std, n_cells=len(tgt_df),
                null_mean=null_mean, null_std=null_std,
                n_null_genes=len(null_per_gene), z=z)


ko_res = zscore_arm("target_ko", "null_ko")
soe_res = zscore_arm("target_soe", "null_soe")

print(
    f"\nGSN KO:   mean={ko_res['mean']:+.4f}, n={ko_res['n_cells']}, "
    f"null={ko_res['null_mean']:+.4f}±{ko_res['null_std']:.4f} "
    f"(n_null={ko_res['n_null_genes']}), z={ko_res['z']:+.2f}"
)
print(
    f"GSN sOE:  mean={soe_res['mean']:+.4f}, n={soe_res['n_cells']}, "
    f"null={soe_res['null_mean']:+.4f}±{soe_res['null_std']:.4f} "
    f"(n_null={soe_res['n_null_genes']}), z={soe_res['z']:+.2f}"
)

# %% — Markdown report
def df_to_md(df_, fmt="{:.4f}"):
    if df_.empty:
        return "_(no rows)_\n"
    cols = list(df_.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df_.iterrows():
        cells = [fmt.format(v) if isinstance(v, float) else str(v) for v in (r[c] for c in cols)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


md = []
md.append("# Phase 4b — GSN Cardiac-Fibroblast Replication: Multi-anchor + soft-OE\n")
md.append(f"**Run:** {datetime.now().isoformat(timespec='seconds')}  ")
md.append("**Script:** `phase4b_gsn_heart_ensemble.py`  ")
md.append("**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  ")
md.append(f"**Data:** TSP1_30 heart cardiac fibroblasts — reuses Phase 4 tokenization  ")
md.append(f"**Pairs:** {N_ANCHORS} anchors × {n_old} old cells = {len(pairs)}  ")
md.append(f"**Specs:** 1 baseline + 2 target + {N_NULL_GENES} null-KO + {N_NULL_GENES} null-sOE = {len(specs)}  ")
md.append(f"**Prompts scored:** {len(df)}\n")

md.append("## Summary\n")
summary = pd.DataFrame([
    {"arm": "KO:GSN", "n_cells": ko_res["n_cells"], "mean_delta_ens": ko_res["mean"],
     "null_mean": ko_res["null_mean"], "null_std": ko_res["null_std"],
     "n_null_genes": ko_res["n_null_genes"], "z_score": ko_res["z"]},
    {"arm": f"sOE{SOFT_OE_BOOST}:GSN", "n_cells": soe_res["n_cells"], "mean_delta_ens": soe_res["mean"],
     "null_mean": soe_res["null_mean"], "null_std": soe_res["null_std"],
     "n_null_genes": soe_res["n_null_genes"], "z_score": soe_res["z"]},
])
md.append(df_to_md(summary))

md.append("## Interpretation\n")
def verdict(res, arm_label, direction_note):
    z = res["z"]
    if np.isnan(z):
        return f"**{arm_label}:** insufficient effective cells — inconclusive.\n"
    if z >= Z_NOTABLE:
        return (f"**{arm_label}: z = {z:+.2f} (≥ +{Z_NOTABLE}).** Rejuvenation signal "
                f"detected. Replication **supported** via {direction_note}.\n")
    if z > 0:
        return (f"**{arm_label}: z = {z:+.2f}.** Direction correct but sub-threshold "
                f"(|z| < {Z_NOTABLE}). Replication **weak**.\n")
    return (f"**{arm_label}: z = {z:+.2f}.** Opposite direction from a rejuvenation "
            f"hit, or effectively null.\n")


md.append(verdict(ko_res, "KO:GSN", "loss-of-function"))
md.append(verdict(soe_res, f"sOE{SOFT_OE_BOOST}:GSN", "gain-of-function"))

md.append("\n## Comparison to Phase 4 (single-anchor)\n")
md.append("Phase 4 (single anchor, N=500 old cells): GSN KO z = **−0.83**, mean_delta = −0.0006. "
          f"Phase 4b multi-anchor averages across {N_ANCHORS} young anchors per old cell, which "
          "should suppress anchor-identity noise if it was masking a real signal. If the KO arm "
          "moves from z ≈ −1 to z ≥ +1, anchor noise was the bottleneck. If it stays null, "
          "the pretraining-only LL readout genuinely does not resolve GSN — next step is to "
          "evaluate with the fine-tuned TimeBetweenCells regression head.\n")

md.append("## Provenance\n")
md.append(f"Outputs in `{SCREEN_DIR}/`:\n")
for f in ("scored_results.csv", "per_old_ensemble.csv"):
    md.append(f"- `{f}`")
md.append("\nRegenerate with `python phase4b_gsn_heart_ensemble.py`.\n")

md_path = OUTPUT_DIR / "phase4b_results.md"
md_path.write_text("\n".join(md))
print(f"Wrote {md_path}")

repo_md = Path("/workspace/phase4b_results.md")
try:
    repo_md.write_text("\n".join(md))
    print(f"Wrote {repo_md}")
except OSError as e:
    print(f"Could not write repo copy: {e}")

print("Done.")
