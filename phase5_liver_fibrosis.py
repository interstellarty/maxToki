# %% [markdown]
# # Phase 5: Liver fibrosis KO screen on Ramachandran 2019
#
# Pivot from aging (LL readout can't resolve, Phase 4) to disease state:
# activated vs quiescent hepatic stellate cells (HSCs). The transcriptional
# gap between activated and quiescent HSCs is large enough to sit above the
# cell-identity noise that dominated Phase 2b, so LL-as-distance-to-quiescent
# should be usable as a screen readout.
#
# Question: Does KO of canonical antifibrotic targets (TGFB1, TGFBR1/2,
# SMAD2/3, CCN2, COL1A1/2/3, ACTA2, PDGFRA/B, LOX/LOXL2, TIMP1/2, FN1) in
# activated HSCs shift their transcriptome toward the quiescent-HSC profile?
# Pro-fibrotic negative controls (SMAD7, MMP1/2/9) should NOT.
#
# Pipeline (mirrors Phase 4 exactly):
#   1. Load Ramachandran tissue.h5ad (converted from tissue.rdata; see
#      convert_ramachandran_to_h5ad.R).
#   2. Filter to HSCs, split by donor condition (cirrhotic=activated,
#      healthy=quiescent).
#   3. Tokenize both splits via `bionemo.maxtoki.data_prep tokenize`.
#   4. Pair: 1 median-length quiescent anchor × up-to-500 activated HSCs.
#   5. Specs: baseline + KO:ENSG for each antifibrotic target + 4 pro-
#      fibrotic neg-ctrls + 50 random null KOs.
#   6. Score, compute ko_took_effect per pair, z-score per spec against null.
#   7. Write phase5_results.md with pass/fail verdict against z >= +2.

# %% — Imports and config
import json
import random
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import anndata as ad
import datasets
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from transformers import AutoModelForCausalLM

warnings.filterwarnings("ignore", category=FutureWarning)

RESOURCES = Path("/workspace/resources")
TOKEN_DICT_PATH = RESOURCES / "token_dictionary_v1.json"
GENE_MEDIAN_PATH = RESOURCES / "gene_median_dictionary_v1.json"
ENSEMBL_MAP_PATH = RESOURCES / "ensembl_mapping_dict_v1.json"
HF_MODEL_PATH = Path("/ptmp/artfi/models/maxtoki-hf/MaxToki-217M-HF")

DATA_DIR = Path("/ptmp/artfi/liver_fibrosis_phase5")
H5AD_PATH = DATA_DIR / "tissue.h5ad"
OUTPUT_DIR = DATA_DIR  # same directory; tokenized outputs go into subdirs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_MAX_TOKENS = 2000
MAX_CELLS = 500
N_NULL_GENES = 50
SEED = 42
Z_NOTABLE = 2.0

ANTIFIBROTIC = [
    "TGFB1", "TGFBR1", "TGFBR2",     # TGF-β axis
    "SMAD2", "SMAD3",                 # canonical signalling
    "CCN2",                           # CTGF, TGF-β downstream effector
    "PDGFRA", "PDGFRB",               # HSC proliferation
    "ACTA2",                          # α-SMA, myofibroblast marker
    "COL1A1", "COL1A2", "COL3A1",     # collagens
    "LOX", "LOXL2",                   # collagen cross-linking
    "TIMP1", "TIMP2",                 # MMP inhibitors (ECM accumulation)
    "FN1",                            # fibronectin
]
PRO_FIBROTIC_NEG_CTRL = [
    "SMAD7",                          # inhibitory Smad; KO should worsen fibrosis
    "MMP1", "MMP2", "MMP9",           # ECM degraders; KO should worsen fibrosis
]

GENE_CATEGORY = {g: "antifibrotic" for g in ANTIFIBROTIC}
GENE_CATEGORY.update({g: "pro_fibrotic_neg_ctrl" for g in PRO_FIBROTIC_NEG_CTRL})

# %% — Load and inspect Ramachandran h5ad
if not H5AD_PATH.exists():
    raise SystemExit(
        f"{H5AD_PATH} not found. Run convert_ramachandran_to_h5ad.R first (see its header)."
    )

adata = sc.read_h5ad(H5AD_PATH)
print(f"Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"obs columns (first 20): {list(adata.obs.columns)[:20]}")

# Auto-detect the cell-type column.
CELL_TYPE_COL = None
for cand in (
    "annotation", "annotation_indepth", "annotation_lineage",
    "cell_type", "cell_ontology_class", "manual_annotation",
):
    if cand in adata.obs.columns:
        CELL_TYPE_COL = cand
        break
if CELL_TYPE_COL is None:
    raise SystemExit(
        f"Could not find a cell-type column in obs. Columns present: {list(adata.obs.columns)}"
    )
print(f"Using cell-type column: {CELL_TYPE_COL!r}")

# Auto-detect the condition column.
CONDITION_COL = None
for cand in (
    "condition", "status", "sample_type", "sample_status",
    "disease", "disease__ontology_label", "health_status",
):
    if cand in adata.obs.columns:
        CONDITION_COL = cand
        break
if CONDITION_COL is None:
    raise SystemExit(
        f"Could not find a condition column in obs. Columns present: {list(adata.obs.columns)}"
    )
print(f"Using condition column: {CONDITION_COL!r}")
print(f"\n{CONDITION_COL} counts:")
print(adata.obs[CONDITION_COL].value_counts())

# %% — Filter to HSCs
ct_lower = adata.obs[CELL_TYPE_COL].astype(str).str.lower()
hsc_mask = ct_lower.str.contains("stellate") | ct_lower.str.contains("hsc")
print(f"\nHSC-matching cells: {hsc_mask.sum()} / {len(adata)}")
if hsc_mask.sum() == 0:
    raise SystemExit(
        f"No HSCs matched. {CELL_TYPE_COL} value counts:\n"
        f"{adata.obs[CELL_TYPE_COL].value_counts()}"
    )
print(adata.obs.loc[hsc_mask, CELL_TYPE_COL].value_counts())

adata_hsc = adata[hsc_mask].copy()

# %% — Split activated (cirrhotic) vs quiescent (healthy)
cond_lower = adata_hsc.obs[CONDITION_COL].astype(str).str.lower()
activated_mask = cond_lower.str.contains("cirrh") | cond_lower.str.contains("fibro") | cond_lower.str.contains("disease")
quiescent_mask = cond_lower.str.contains("healthy") | cond_lower.str.contains("normal") | cond_lower.str.contains("control")

print(f"\nActivated HSCs (cirrhotic/fibrotic/diseased): {activated_mask.sum()}")
print(f"Quiescent HSCs (healthy/normal/control):     {quiescent_mask.sum()}")
if activated_mask.sum() == 0 or quiescent_mask.sum() == 0:
    print("Condition value counts within HSCs:")
    print(adata_hsc.obs[CONDITION_COL].value_counts())
    raise SystemExit("Empty activated or quiescent HSC pool — adjust condition detection.")

adata_activated = adata_hsc[activated_mask].copy()
adata_quiescent = adata_hsc[quiescent_mask].copy()

# %% — Prepare h5ads for tokenization (same mechanics as Phase 1/4)
with open(TOKEN_DICT_PATH) as f:
    _token_dict_probe = json.load(f)
with open(ENSEMBL_MAP_PATH) as f:
    _map_probe = json.load(f)

_probe = adata_hsc.var
has_ensembl_col = "ensembl_id" in _probe.columns
has_symbol_col = "gene_symbol" in _probe.columns
idx_looks_ensembl = str(_probe.index[0]).startswith("ENSG")

_hits = {}
if has_ensembl_col:
    _hits["ensembl_id"] = _probe["ensembl_id"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
if has_symbol_col:
    _hits["gene_symbol"] = _probe["gene_symbol"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
_hits["index"] = _probe.index.astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
# Also try direct vocab hits on the index (in case it is already ENSG).
if idx_looks_ensembl:
    _hits["index_direct"] = _probe.index.astype(str).str.upper().isin(_token_dict_probe).sum()

print(f"\nToken-dict hits per candidate var column: {_hits}")
_best_col = max(_hits, key=_hits.get)
print(f"Best column for tokenization: {_best_col!r} ({_hits[_best_col]} hits)")
if _hits[_best_col] < 1000:
    raise SystemExit(f"No column hits the vocab sufficiently: {_hits}")

USE_INDEX_AS_ENSEMBL = _best_col in ("index", "index_direct")

# n_counts for the tokenizer.
if "n_counts" not in adata_hsc.obs.columns:
    adata_activated.obs["n_counts"] = np.array(adata_activated.X.sum(axis=1)).flatten()
    adata_quiescent.obs["n_counts"] = np.array(adata_quiescent.X.sum(axis=1)).flatten()

# Donor column (for time_group grouping in data_prep). Fall back to condition.
donor_col = None
for cand in ("donor", "donor_id", "patient", "patient_id", "sample", "sample_id"):
    if cand in adata_hsc.obs.columns:
        donor_col = cand
        break
if donor_col is None:
    donor_col = CONDITION_COL
print(f"Using donor/sample column for time_group: {donor_col!r}")

for adata_split in (adata_activated, adata_quiescent):
    adata_split.obs["time"] = 0
    adata_split.obs["unique_cell_id"] = adata_split.obs_names.astype(str)
    adata_split.obs["time_group"] = adata_split.obs[donor_col].astype(str)
    if _best_col == "gene_symbol":
        adata_split.var["ensembl_id"] = adata_split.var["gene_symbol"].astype(str)
    elif _best_col in ("index", "index_direct"):
        adata_split.var["ensembl_id"] = adata_split.var.index.astype(str)

activated_dir = OUTPUT_DIR / "input_activated"
quiescent_dir = OUTPUT_DIR / "input_quiescent"
activated_dir.mkdir(exist_ok=True)
quiescent_dir.mkdir(exist_ok=True)
activated_h5ad = activated_dir / "hsc_activated.h5ad"
quiescent_h5ad = quiescent_dir / "hsc_quiescent.h5ad"
adata_activated.write_h5ad(activated_h5ad)
adata_quiescent.write_h5ad(quiescent_h5ad)
print(f"Wrote {len(adata_activated)} activated, {len(adata_quiescent)} quiescent HSCs.")

# %% — Tokenize both splits
for label, input_dir in [("activated", activated_dir), ("quiescent", quiescent_dir)]:
    out_dir = OUTPUT_DIR / f"tokenized_{label}"
    if any(out_dir.glob("*.dataset")):
        print(f"Tokenized {label} dataset already exists at {out_dir} — skipping.")
        continue
    out_dir.mkdir(exist_ok=True)
    cmd = [
        "python", "-m", "bionemo.maxtoki.data_prep", "tokenize",
        "--data-directory", str(input_dir),
        "--output-directory", str(out_dir),
        "--output-prefix", f"hsc_{label}",
        "--token-dictionary-file", str(TOKEN_DICT_PATH),
        "--gene-median-file", str(GENE_MEDIAN_PATH),
        "--gene-mapping-file", str(ENSEMBL_MAP_PATH),
        "--nproc", "4",
        "--model-input-size", "4096",
    ]
    if USE_INDEX_AS_ENSEMBL:
        cmd.append("--use-h5ad-index")
    print(f"Tokenizing {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise SystemExit(f"Tokenization failed for {label}.")

ds_activated = datasets.load_from_disk(str(next((OUTPUT_DIR / "tokenized_activated").glob("*.dataset"))))
ds_quiescent = datasets.load_from_disk(str(next((OUTPUT_DIR / "tokenized_quiescent").glob("*.dataset"))))
print(f"Tokenized: {len(ds_activated)} activated, {len(ds_quiescent)} quiescent")

# %% — Tokenizer + CellPair construction
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


quiescent_lengths = [len(r["input_ids"]) for r in ds_quiescent]
anchor_idx = int(np.argsort(quiescent_lengths)[len(quiescent_lengths) // 2])
anchor_tokens = _truncate(ds_quiescent[anchor_idx]["input_ids"])
print(f"Quiescent anchor: idx={anchor_idx}, len={len(anchor_tokens)}")

n_activated = min(len(ds_activated), MAX_CELLS)
pairs = []
for i in range(n_activated):
    pairs.append(CellPair(
        young_tokens=anchor_tokens,
        old_tokens=_truncate(ds_activated[i]["input_ids"]),
        cell_id=f"activated_hsc:{i}",
        metadata={"cell_type": "activated_hsc"},
    ))
print(f"Built {len(pairs)} pairs (1 quiescent anchor × {n_activated} activated HSCs).")

# %% — KO specs: baseline + antifibrotic targets + pro-fibrotic neg ctrls + null
specs = [make_noop_spec()]
target_ensgs = {}       # eid -> gene_symbol (antifibrotic)
neg_ctrl_ensgs = {}     # eid -> gene_symbol (pro-fibrotic neg-ctrl)
skipped = []

for gene in ANTIFIBROTIC:
    try:
        eid = symbol_to_ensembl(gene)
    except KeyError:
        skipped.append(gene)
        continue
    if eid not in token_dict:
        skipped.append(gene)
        continue
    specs.append(make_knockout_spec(tokenizer, eid))
    target_ensgs[eid] = gene
    print(f"  KO spec: {gene} -> {eid}")

for gene in PRO_FIBROTIC_NEG_CTRL:
    try:
        eid = symbol_to_ensembl(gene)
    except KeyError:
        skipped.append(gene)
        continue
    if eid not in token_dict:
        skipped.append(gene)
        continue
    specs.append(make_knockout_spec(tokenizer, eid))
    neg_ctrl_ensgs[eid] = gene
    print(f"  KO spec (neg-ctrl): {gene} -> {eid}")

if skipped:
    print(f"Skipped (not in vocab): {skipped}")

_rng = random.Random(SEED)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_exclude = set(target_ensgs) | set(neg_ctrl_ensgs)
_null_pool = [e for e in _vocab_ensgs if e not in _exclude]
null_ensgs = _rng.sample(_null_pool, N_NULL_GENES)
for eid in null_ensgs:
    specs.append(make_knockout_spec(tokenizer, eid))

print(
    f"\nSpecs: 1 baseline + {len(target_ensgs)} antifibrotic + "
    f"{len(neg_ctrl_ensgs)} neg-ctrl + {N_NULL_GENES} null = {len(specs)} total"
)

# %% — Build screen dataset and score
SCREEN_DIR = OUTPUT_DIR / "screen_hsc_fibrosis_v1"
ds_path, manifest_path = build_screen_dataset_scoring(
    tokenizer, pairs, specs, output_dir=SCREEN_DIR, model_input_size=4096,
)
manifest = pd.read_csv(manifest_path)
print(f"Screen: {len(manifest)} prompts -> {SCREEN_DIR}")

print(f"Loading MaxToki-217M-HF from {HF_MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    str(HF_MODEL_PATH), torch_dtype=torch.bfloat16,
).to("cuda").eval()
print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

print("Scoring...")
df = score_screen(model, ds_path, manifest_path, device="cuda")
df.to_csv(SCREEN_DIR / "scored_results.csv", index=False)

# %% — ko_took_effect + spec classification
baseline_lengths = df[df["spec_name"] == "baseline"].set_index("pair_idx")["prompt_length"]
df["ko_took_effect"] = df["pair_idx"].map(baseline_lengths) != df["prompt_length"]

effect_rates = (
    df[df["spec_name"] != "baseline"]
    .groupby("spec_name")["ko_took_effect"]
    .mean()
)
effect_rates.to_csv(SCREEN_DIR / "ko_effect_rates.csv", header=["effect_rate"])

target_spec_names = {f"KO:{e}" for e in target_ensgs}
neg_ctrl_spec_names = {f"KO:{e}" for e in neg_ctrl_ensgs}
null_spec_names = {f"KO:{e}" for e in null_ensgs}

df["spec_class"] = "baseline"
df.loc[df["spec_name"].isin(null_spec_names), "spec_class"] = "null"
df.loc[df["spec_name"].isin(target_spec_names), "spec_class"] = "antifibrotic"
df.loc[df["spec_name"].isin(neg_ctrl_spec_names), "spec_class"] = "neg_ctrl"

ensg_to_symbol = {**target_ensgs, **neg_ctrl_ensgs}
df["gene_symbol"] = df["spec_name"].str.replace("KO:", "", regex=False).map(ensg_to_symbol)
df["gene_category"] = df["gene_symbol"].map(GENE_CATEGORY)

# %% — Null + per-target stats (effective KOs only)
effective = df[df["ko_took_effect"]]

null_per_gene = (
    effective[effective["spec_class"] == "null"]
    .groupby("spec_name")["delta_vs_baseline"].mean()
)
null_mean = null_per_gene.mean()
null_std = null_per_gene.std()
n_null_genes = len(null_per_gene)
print(f"\nNull: mean={null_mean:+.4f}, std={null_std:.4f}, n_effective_genes={n_null_genes}")

per_target = (
    effective[effective["spec_class"].isin({"antifibrotic", "neg_ctrl"})]
    .groupby(["spec_name", "spec_class", "gene_symbol", "gene_category"])
    .agg(
        mean_delta=("delta_vs_baseline", "mean"),
        std_delta=("delta_vs_baseline", "std"),
        n_effective=("delta_vs_baseline", "count"),
    )
    .reset_index()
)
per_target["null_mean"] = null_mean
per_target["null_std"] = null_std
per_target["z_score"] = (per_target["mean_delta"] - null_mean) / null_std
per_target = per_target.sort_values("z_score", ascending=False)
per_target.to_csv(SCREEN_DIR / "per_target_zscored.csv", index=False)
print("\nTop 10 by z-score:")
print(per_target.head(10).to_string())

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
md.append("# Phase 5 — Liver Fibrosis KO Screen: Results (Ramachandran 2019)\n")
md.append(f"**Run:** {datetime.now().isoformat(timespec='seconds')}  ")
md.append("**Script:** `phase5_liver_fibrosis.py`  ")
md.append("**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  ")
md.append(
    f"**Data:** Ramachandran 2019 liver — {len(adata_activated)} activated (cirrhotic) / "
    f"{len(adata_quiescent)} quiescent (healthy) HSCs  "
)
md.append(f"**Pairs:** 1 quiescent anchor × {n_activated} activated = {len(pairs)}  ")
md.append(
    f"**Specs:** 1 baseline + {len(target_ensgs)} antifibrotic + {len(neg_ctrl_ensgs)} "
    f"pro-fibrotic neg-ctrl + {N_NULL_GENES} null = {len(specs)}  "
)
md.append(f"**Prompts scored:** {len(df)}\n")
if skipped:
    md.append(f"**Skipped (not in MaxToki vocab):** {', '.join(skipped)}\n")

md.append("## Null distribution (50 random-gene KOs, effective only)\n")
md.append(f"- Effective null genes: {n_null_genes}\n")
md.append(f"- Null mean delta: {null_mean:+.4f}\n")
md.append(f"- Null std: {null_std:.4f}\n")

md.append("## KO effect rates (fraction of activated HSCs where KO dropped a token)\n")
md.append("Low effect rate → gene rarely in top-2000 tokens → z-score rests on few cells.\n")
effect_df = effect_rates.reset_index()
effect_df["gene_symbol"] = effect_df["spec_name"].str.replace("KO:", "", regex=False).map(ensg_to_symbol)
effect_df["gene_category"] = effect_df["gene_symbol"].map(GENE_CATEGORY)
effect_df = effect_df[effect_df["gene_symbol"].notna()].sort_values("effect_rate", ascending=False)
md.append(df_to_md(effect_df[["gene_symbol", "gene_category", "effect_rate"]], fmt="{:.2f}"))

md.append("## Z-scored per-target results\n")
md.append(
    "`delta_vs_baseline > 0` = perturbed activated HSC looks more like the quiescent "
    f"anchor. `z >= +{Z_NOTABLE}` is notable. Antifibrotic KOs should score positive; "
    "pro-fibrotic neg-ctrl KOs should not.\n"
)
pz = per_target[[
    "gene_symbol", "gene_category", "spec_class",
    "mean_delta", "n_effective", "null_mean", "null_std", "z_score",
]]
md.append(df_to_md(pz))

md.append("## Key findings\n")
notable_antifib = per_target[
    (per_target["spec_class"] == "antifibrotic") & (per_target["z_score"] >= Z_NOTABLE)
]
notable_negctrl = per_target[
    (per_target["spec_class"] == "neg_ctrl") & (per_target["z_score"].abs() >= Z_NOTABLE)
]

if len(notable_antifib) > 0:
    md.append(f"### Antifibrotic hits passing z >= +{Z_NOTABLE}\n")
    md.append(df_to_md(notable_antifib[[
        "gene_symbol", "mean_delta", "n_effective", "z_score",
    ]]))
    md.append(
        f"**{len(notable_antifib)} antifibrotic KO(s) passed z >= +{Z_NOTABLE}.** "
        "Methodological validation of the LL-as-distance-to-quiescent framing: "
        "canonical antifibrotic targets behave as literature predicts.\n"
    )
else:
    md.append(
        f"**No antifibrotic KO passed z >= +{Z_NOTABLE}.** "
        "Either the LL-as-distance-to-quiescent framing is underpowered at this scale "
        "(next step: Phase 5b with multi-anchor ensembling, Phase 4b's trick), or the "
        "identity-saturation issue from Phase 2b extends to disease-state as well.\n"
    )

if len(notable_negctrl) > 0:
    md.append(f"### Pro-fibrotic neg-ctrl KOs passing |z| >= {Z_NOTABLE} (should be none)\n")
    md.append(df_to_md(notable_negctrl[[
        "gene_symbol", "mean_delta", "n_effective", "z_score",
    ]]))
    md.append(
        f"**{len(notable_negctrl)} pro-fibrotic neg-ctrl(s) significant.** "
        "If direction matches (z >= +2), the null model is too permissive. If opposite "
        "(z <= -2), the screen is calibrated but the neg-ctrl mechanism may be atypical.\n"
    )
else:
    md.append(f"No pro-fibrotic neg-ctrl KO passed |z| >= {Z_NOTABLE}. Negative controls behave as expected.\n")

md.append("## Provenance\n")
md.append(f"Outputs in `{SCREEN_DIR}/`:\n")
for fname in ("scored_results.csv", "ko_effect_rates.csv", "per_target_zscored.csv"):
    md.append(f"- `{fname}`")
md.append("\nRegenerate with `python phase5_liver_fibrosis.py`.\n")

md_path = OUTPUT_DIR / "phase5_results.md"
md_path.write_text("\n".join(md))
print(f"\nWrote {md_path}")

repo_md = Path("/workspace/phase5_results.md")
try:
    repo_md.write_text("\n".join(md))
    print(f"Wrote {repo_md}")
except OSError as e:
    print(f"Could not write repo copy: {e}")

print("Done.")
