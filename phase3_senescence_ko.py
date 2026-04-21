# %% [markdown]
# # Phase 3: Senescence-Gene Knockout Screen
#
# Phase 2 and 2b showed MaxToki can't read a rejuvenation signal out of TF
# overexpression — promoting any TF (even mildly) displaces cell-identity
# markers and makes the perturbed cell look less like its own young anchor.
# This phase uses the opposite operator: **knockout**.
#
# KO is in-distribution (simply drops a token) and, for senescence drivers
# and SASP components that are *upregulated* in old cells, should test the
# age axis directly. Longevity genes (SIRT1/6, FOXO3) serve as negative
# controls — their KO should not rejuvenate, and might hurt.
#
# Readout: `delta_vs_baseline` per (spec, cell_type), z-scored against a null
# distribution of 50 random-gene KOs, restricted to cells where the KO
# actually dropped a token (`ko_took_effect`).
#
# ## Scale-up rationale (vs. the original Phase 3 run at MAX_CELLS_PER_TYPE=50)
#
# The initial run surfaced CDKN1A (p21) KO in hepatocytes at z = +1.93 —
# directionally correct but just below the |z| ≥ 2 threshold — on only
# **n = 3 effective cells** (p21 is in the top-2000 tokens of only 6% of old
# hepatocytes). Every senescence driver pointed positive and every longevity
# gene pointed neutral-to-negative, but no hit crossed significance.
#
# To resolve whether the top hit is a real signal or small-n noise, this
# version bumps `MAX_CELLS_PER_TYPE` from 50 → 500. TSP has hundreds of old
# hepatocytes available; scaling the sample keeps the baseline pair count up
# (so 6% × 500 = ~30 effective cells for CDKN1A, vs. 3 previously) without
# changing any other part of the pipeline. If CDKN1A moves to z ≥ 3 it is a
# solid hit; if it stays around z ≈ 1–2 it is likely noise. Runtime grows
# roughly linearly with pair count — expect ~3 hours on A100 instead of ~20
# minutes, so launch via sbatch or tmux.
#
# Outputs are written to `/ptmp/$USER/liver_screen_phase3b/` and
# `phase3b_results.md` (separate from the original Phase 3 outputs so the
# n=50 results and their manual interpretation are preserved).

# %% — Imports and config
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RESOURCES = Path("/workspace/resources")
TOKEN_DICT_PATH = RESOURCES / "token_dictionary_v1.json"
ENSEMBL_MAP_PATH = RESOURCES / "ensembl_mapping_dict_v1.json"
HF_MODEL_PATH = Path("/ptmp/artfi/models/maxtoki-hf/MaxToki-217M-HF")
PHASE1_DIR = Path("/ptmp/artfi/liver_screen")
OUTPUT_DIR = Path("/ptmp/artfi/liver_screen_phase3b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_TYPE_COL = "cell_ontology_class"
CELL_MAX_TOKENS = 2000
MAX_CELLS_PER_TYPE = 500  # bumped from 50 to resolve n=3 CDKN1A hit
N_NULL_GENES = 50
SEED = 42

# Z-score threshold for "notable" hits in the markdown summary.
Z_NOTABLE = 2.0

# Senescence drivers (core cell-cycle arrest program; KO -> rejuvenation if model captures it).
SENESCENCE_DRIVERS = ["CDKN2A", "CDKN1A", "TP53", "RB1"]

# SASP components (secreted pro-inflammatory / ECM-remodeling; KO -> rejuvenation).
SASP_GENES = ["IL6", "CXCL8", "CCL2", "SERPINE1", "MMP3", "TNF"]

# Senescence markers upregulated in aging.
AGING_MARKERS = ["GLB1", "MDM2"]

# Negative controls: longevity genes; KO should NOT rejuvenate (ideally anti-rejuvenate).
LONGEVITY_NEG_CONTROLS = ["SIRT1", "SIRT6", "FOXO3"]

POSITIVE_KO_GENES = SENESCENCE_DRIVERS + SASP_GENES + AGING_MARKERS
NEG_CONTROL_GENES = LONGEVITY_NEG_CONTROLS

GENE_CATEGORY = {g: "senescence_driver" for g in SENESCENCE_DRIVERS}
GENE_CATEGORY.update({g: "sasp" for g in SASP_GENES})
GENE_CATEGORY.update({g: "aging_marker" for g in AGING_MARKERS})
GENE_CATEGORY.update({g: "longevity_neg_ctrl" for g in NEG_CONTROL_GENES})

# %% — Load Phase 1 tokenized datasets and adata splits
import anndata as ad
import datasets

ds_young_root = PHASE1_DIR / "tokenized_young"
ds_old_root = PHASE1_DIR / "tokenized_old"
adata_young_path = PHASE1_DIR / "input_young" / "ts_liver_young.h5ad"
adata_old_path = PHASE1_DIR / "input_old" / "ts_liver_old.h5ad"

if not all(p.exists() for p in (ds_young_root, ds_old_root, adata_young_path, adata_old_path)):
    raise SystemExit(
        "Phase 1 outputs missing. Run phase1_liver_screen.py first to produce "
        f"tokenized datasets at {PHASE1_DIR}."
    )

adata_young = ad.read_h5ad(adata_young_path)
adata_old = ad.read_h5ad(adata_old_path)
ds_young = datasets.load_from_disk(str(next(ds_young_root.glob("*.dataset"))))
ds_old = datasets.load_from_disk(str(next(ds_old_root.glob("*.dataset"))))
print(f"Loaded: {len(ds_young)} young / {len(ds_old)} old cells.")

# %% — Build CellPairs (same logic as Phase 2 / 2b)
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
from bionemo.maxtoki.perturb import (
    CellPair, build_screen_dataset_scoring, score_screen,
    make_noop_spec, make_knockout_spec,
)

with open(TOKEN_DICT_PATH) as f:
    token_dict = json.load(f)
with open(ENSEMBL_MAP_PATH) as f:
    ensembl_map = json.load(f)
tokenizer = MaxTokiTokenizer(token_dict)


def symbol_to_ensembl(symbol):
    eid = ensembl_map.get(symbol.upper())
    if eid is None:
        raise KeyError(f"Gene {symbol!r} not found in ensembl_mapping_dict_v1.json")
    return eid


def _truncate(tokens):
    if len(tokens) <= CELL_MAX_TOKENS:
        return list(tokens)
    bos = [tokens[0]] if tokens and tokens[0] == tokenizer.bos_id else []
    eos = [tokens[-1]] if tokens and tokens[-1] == tokenizer.eos_id else []
    body = tokens[len(bos): len(tokens) - len(eos)]
    body = body[: CELL_MAX_TOKENS - len(bos) - len(eos)]
    return [*bos, *body, *eos]


pairs = []
pairs_per_type = {}
for ct in adata_old.obs[CELL_TYPE_COL].unique():
    young_ct = ds_young.filter(
        lambda x, idx: adata_young.obs.iloc[idx][CELL_TYPE_COL] == ct, with_indices=True,
    ) if len(ds_young) > 0 else None
    old_ct = ds_old.filter(
        lambda x, idx: adata_old.obs.iloc[idx][CELL_TYPE_COL] == ct, with_indices=True,
    ) if len(ds_old) > 0 else None
    if not young_ct or not old_ct or len(young_ct) == 0 or len(old_ct) == 0:
        continue

    young_lengths = [len(r["input_ids"]) for r in young_ct]
    anchor_idx = int(np.argsort(young_lengths)[len(young_lengths) // 2])
    anchor_tokens = _truncate(young_ct[anchor_idx]["input_ids"])

    n_old = min(len(old_ct), MAX_CELLS_PER_TYPE)
    for i in range(n_old):
        pairs.append(CellPair(
            young_tokens=anchor_tokens,
            old_tokens=_truncate(old_ct[i]["input_ids"]),
            cell_id=f"{ct}:{i}",
            metadata={"cell_type": ct},
        ))
    pairs_per_type[ct] = n_old
    print(f"  {ct}: 1 young anchor, {n_old} old cells")

print(f"Total pairs: {len(pairs)}")

# %% — Build KO specs: positives + negative controls + null
specs = [make_noop_spec()]
positive_ko_ensgs = {}   # eid -> gene_symbol
neg_control_ensgs = {}
skipped = []

for gene in POSITIVE_KO_GENES + NEG_CONTROL_GENES:
    try:
        eid = symbol_to_ensembl(gene)
    except KeyError:
        skipped.append(gene)
        continue
    if eid not in token_dict:
        skipped.append(gene)
        continue
    specs.append(make_knockout_spec(tokenizer, eid))
    if gene in POSITIVE_KO_GENES:
        positive_ko_ensgs[eid] = gene
    else:
        neg_control_ensgs[eid] = gene
    print(f"  KO spec: {gene} -> {eid}")

if skipped:
    print(f"  Skipped (not in vocab): {skipped}")

# Null: 50 random-gene KOs, excluded from the positive/neg control sets.
_rng = random.Random(SEED)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_exclude = set(positive_ko_ensgs) | set(neg_control_ensgs)
_null_pool = [e for e in _vocab_ensgs if e not in _exclude]
null_ensgs = _rng.sample(_null_pool, N_NULL_GENES)
for eid in null_ensgs:
    specs.append(make_knockout_spec(tokenizer, eid))

print(
    f"\nSpecs: 1 baseline + {len(positive_ko_ensgs)} positive-KO + "
    f"{len(neg_control_ensgs)} neg-ctrl + {N_NULL_GENES} null = {len(specs)} total"
)

# %% — Build screen dataset
SCREEN_DIR = OUTPUT_DIR / "screen_senescence_ko_scaled_v1"
ds_path, manifest_path = build_screen_dataset_scoring(
    tokenizer,
    pairs,
    specs,
    output_dir=SCREEN_DIR,
    model_input_size=4096,
)
manifest = pd.read_csv(manifest_path)
print(f"Screen dataset: {len(manifest)} prompts -> {SCREEN_DIR}")

# %% — Score screen
import torch
from transformers import AutoModelForCausalLM

print("Loading MaxToki-217M-HF...")
model = AutoModelForCausalLM.from_pretrained(
    str(HF_MODEL_PATH),
    torch_dtype=torch.bfloat16,
).to("cuda").eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

print("Scoring screen...")
df = score_screen(model, ds_path, manifest_path, device="cuda")
df.to_csv(SCREEN_DIR / "scored_results.csv", index=False)

# %% — Diagnostic: did each KO actually drop a token?
baseline_lengths = df[df["spec_name"] == "baseline"].set_index("pair_idx")["prompt_length"]
df["ko_took_effect"] = df["pair_idx"].map(baseline_lengths) != df["prompt_length"]

effect_rates = (
    df[df["spec_name"] != "baseline"]
    .groupby(["spec_name", "cell_type"])["ko_took_effect"]
    .mean()
    .unstack()
)
effect_rates.to_csv(SCREEN_DIR / "ko_effect_rates.csv")

# %% — Annotate spec classes and compute z-scores against null
positive_spec_names = {f"KO:{e}" for e in positive_ko_ensgs}
neg_control_spec_names = {f"KO:{e}" for e in neg_control_ensgs}
null_spec_names = {f"KO:{e}" for e in null_ensgs}

df["spec_class"] = "baseline"
df.loc[df["spec_name"].isin(null_spec_names), "spec_class"] = "null"
df.loc[df["spec_name"].isin(positive_spec_names), "spec_class"] = "positive"
df.loc[df["spec_name"].isin(neg_control_spec_names), "spec_class"] = "neg_control"

# Gene symbol + category columns (useful in the markdown output).
ensg_to_symbol = {**positive_ko_ensgs, **neg_control_ensgs}
df["gene_symbol"] = df["spec_name"].str.replace("KO:", "", regex=False).map(ensg_to_symbol)
df["gene_category"] = df["gene_symbol"].map(GENE_CATEGORY)

# Null stats computed on effective KOs only, per cell type.
effective = df[df["ko_took_effect"]]
null_per_gene = (
    effective[effective["spec_class"] == "null"]
    .groupby(["cell_type", "spec_name"])["delta_vs_baseline"]
    .mean()
)
null_stats = null_per_gene.groupby("cell_type").agg(["mean", "std", "count"])
null_stats.to_csv(SCREEN_DIR / "null_stats.csv")

positive_per_gene = (
    effective[effective["spec_class"].isin({"positive", "neg_control"})]
    .groupby(["cell_type", "spec_name", "spec_class", "gene_symbol", "gene_category"])
    .agg(
        mean_delta=("delta_vs_baseline", "mean"),
        std_delta=("delta_vs_baseline", "std"),
        n_effective=("delta_vs_baseline", "count"),
    )
    .reset_index()
)
positive_per_gene["null_mean"] = positive_per_gene["cell_type"].map(null_stats["mean"])
positive_per_gene["null_std"] = positive_per_gene["cell_type"].map(null_stats["std"])
positive_per_gene["z_score"] = (
    (positive_per_gene["mean_delta"] - positive_per_gene["null_mean"]) / positive_per_gene["null_std"]
)
positive_per_gene = positive_per_gene.sort_values("z_score", ascending=False)
positive_per_gene.to_csv(SCREEN_DIR / "positive_zscored.csv", index=False)

# %% — Build a markdown results report
from datetime import datetime


def df_to_md(df_to_render: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a pipe-style markdown table (no external deps)."""
    if df_to_render.empty:
        return "_(no rows)_\n"
    cols = list(df_to_render.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df_to_render.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


md: list[str] = []
md.append("# Phase 3 — Senescence-Gene Knockout Screen: Results\n")
md.append(f"**Run:** {datetime.now().isoformat(timespec='seconds')}  ")
md.append("**Script:** `phase3_senescence_ko.py`  ")
md.append("**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  ")
md.append(f"**Data:** TSP1_30 liver — {sum(pairs_per_type.values())} young–old pairs across "
          f"{len(pairs_per_type)} cell types  ")
md.append(f"**Specs:** 1 baseline + {len(positive_ko_ensgs)} senescence/SASP/aging KOs + "
          f"{len(neg_control_ensgs)} longevity-negative-control KOs + {N_NULL_GENES} null KOs = "
          f"{len(specs)} total  ")
md.append(f"**Prompts scored:** {len(df)}\n")
if skipped:
    md.append(f"**Skipped (not in MaxToki vocab):** {', '.join(skipped)}\n")

# Pairs per cell type.
md.append("## Pairs per cell type\n")
md.append(df_to_md(pd.DataFrame([
    {"cell_type": ct, "n_old_cells": n} for ct, n in pairs_per_type.items()
])))

# ko_took_effect rates: fraction of old cells where each KO dropped a token.
md.append("## KO effect rates (fraction of old cells where the KO dropped a token)\n")
md.append(
    "A gene with low effect rate is absent from most old cells' top-"
    f"{CELL_MAX_TOKENS} tokens — any z-score based on it rests on few cells.\n"
)
effect_table = effect_rates.copy()
effect_table = effect_table.reset_index()
effect_table["gene_symbol"] = effect_table["spec_name"].str.replace("KO:", "", regex=False).map(ensg_to_symbol)
effect_table["gene_category"] = effect_table["gene_symbol"].map(GENE_CATEGORY)
# Drop null-gene rows from the markdown table to keep it focused.
effect_table = effect_table[effect_table["gene_symbol"].notna()]
effect_cols = ["gene_symbol", "gene_category"] + [c for c in effect_table.columns
                                                    if c not in {"gene_symbol", "gene_category", "spec_name"}]
md.append(df_to_md(effect_table[effect_cols], float_fmt="{:.2f}"))

# Null distribution.
md.append("## Null distribution (50 random-gene KOs, effective KOs only)\n")
null_md = null_stats.reset_index()
null_md.columns = ["cell_type", "null_mean", "null_std", "n_effective_genes"]
md.append(df_to_md(null_md))

# Full z-score table for positives + neg controls.
md.append("## Z-scored positives and negative controls\n")
md.append(
    "`delta_vs_baseline > 0` = perturbed old cell looks more like the young anchor "
    "(rejuvenation signal). `|z| > 2` is notable.\n"
)
pz = positive_per_gene[[
    "gene_symbol", "gene_category", "cell_type", "spec_class",
    "mean_delta", "n_effective", "null_mean", "null_std", "z_score",
]]
md.append(df_to_md(pz))

# Key findings: anything above Z_NOTABLE, in either direction, separated by class.
notable_pos = positive_per_gene[
    (positive_per_gene["spec_class"] == "positive") & (positive_per_gene["z_score"].abs() >= Z_NOTABLE)
]
notable_neg_ctrl = positive_per_gene[
    (positive_per_gene["spec_class"] == "neg_control") & (positive_per_gene["z_score"].abs() >= Z_NOTABLE)
]

md.append("## Key findings\n")
if len(notable_pos) > 0:
    md.append(f"### Positive controls passing |z| ≥ {Z_NOTABLE}\n")
    md.append(df_to_md(notable_pos[[
        "gene_symbol", "gene_category", "cell_type", "mean_delta", "n_effective", "z_score"
    ]]))
    rejuvenators = notable_pos[notable_pos["z_score"] > 0]
    if len(rejuvenators) > 0:
        md.append(
            f"**{len(rejuvenators)} senescence/SASP/aging-marker KO(s) show a rejuvenation "
            f"signal (z ≥ +{Z_NOTABLE}).** This is the real-deal finding: KO of these genes "
            "makes the old cell's transcriptome look more like its young counterpart.\n"
        )
    anti = notable_pos[notable_pos["z_score"] < 0]
    if len(anti) > 0:
        md.append(
            f"**{len(anti)} positive-control KO(s) show z ≤ −{Z_NOTABLE}** — unexpected; "
            "KO pushes the cell further from young. Worth inspecting for off-target rank "
            "disruption.\n"
        )
else:
    md.append(f"No positive-control KO passed |z| ≥ {Z_NOTABLE} in any cell type. "
              "MaxToki does not distinguish young vs. old liver cells along the canonical "
              "senescence axis at 217M scale.\n")

if len(notable_neg_ctrl) > 0:
    md.append(f"### Negative controls passing |z| ≥ {Z_NOTABLE} (should ideally be none)\n")
    md.append(df_to_md(notable_neg_ctrl[[
        "gene_symbol", "gene_category", "cell_type", "mean_delta", "n_effective", "z_score"
    ]]))
else:
    md.append(f"No longevity-gene KO passed |z| ≥ {Z_NOTABLE}. Negative controls behave as "
              "expected.\n")

md.append("## Provenance\n")
md.append(f"Outputs in `{SCREEN_DIR}/`:\n")
for fname in ["scored_results.csv", "ko_effect_rates.csv", "null_stats.csv", "positive_zscored.csv"]:
    md.append(f"- `{fname}`")
md.append(f"\nRegenerate with `python phase3_senescence_ko.py` on branch "
          f"`claude/continue-previous-session-E56FX`.\n")

md_path = OUTPUT_DIR / "phase3b_results.md"
md_path.write_text("\n".join(md))
print(f"\nWrote markdown summary to {md_path}")

# Also drop a copy next to the repo for easy git-add.
repo_md = Path("/workspace/phase3b_results.md")
try:
    repo_md.write_text("\n".join(md))
    print(f"Wrote copy to {repo_md} (commit from host to preserve across runs).")
except OSError as e:
    print(f"Could not write {repo_md}: {e}")

print("\nDone.")
