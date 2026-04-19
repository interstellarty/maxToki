# %% [markdown]
# # Phase 2b: Soft-OE Rejuvenation Screen
#
# Phase 2 found that promoting TFs to rank 0 (`make_overexpression_spec`) is
# out-of-distribution — OSK/OSKM combos got the most negative z-scores
# (OSK_L in HSC: z=-2.46, OSKM in HSC: z=-2.0). This is likely an artifact of
# unphysiological rank-0 pinning, since TFs normally sit mid-rank.
#
# This phase re-runs with `make_soft_overexpression_spec`, which shifts a gene
# up by `boost_ranks` positions rather than pinning it to rank 0. Includes:
#
# - Single-TF soft OE at K=50 for 8 candidates (Yamanaka, liver-identity, longevity)
# - OSK and OSKM combo soft OE at K=50
# - **Boost-rank sweep** on OSKM: K ∈ {10, 25, 50, 100, 200, 500, 2000}.
#   At K=2000 soft OE collapses to hard OE, so we bracket both the
#   physiological regime and the OOD regime in a single run.
# - 50 random single-gene soft OEs at K=50 as null.
#
# Run inside the Apptainer container on a GPU node. Reuses Phase 1's
# tokenized datasets at `/ptmp/artfi/liver_screen/tokenized_{young,old}`.

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
OUTPUT_DIR = Path("/ptmp/artfi/liver_screen_phase2b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_TYPE_COL = "cell_ontology_class"
CELL_MAX_TOKENS = 2000
MAX_CELLS_PER_TYPE = 50
N_NULL_GENES = 50
SEED = 42
DEFAULT_BOOST_RANKS = 50
OSKM_SWEEP_RANKS = [10, 25, 50, 100, 200, 500, 2000]

SINGLE_TFS = [
    "POU5F1", "SOX2", "KLF4", "MYC",   # Yamanaka
    "HNF4A",                             # hepatocyte master
    "NANOG",                             # pluripotency
    "SIRT1", "SIRT6",                    # longevity — SIRT6 was top positive in Phase 2
]
COMBO_TFS = {
    "OSK":  ["POU5F1", "SOX2", "KLF4"],
    "OSKM": ["POU5F1", "SOX2", "KLF4", "MYC"],
}

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

# %% — Build CellPairs (same logic as Phase 2)
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
from bionemo.maxtoki.perturb import (
    CellPair, build_screen_dataset_scoring, score_screen,
    make_noop_spec, make_soft_overexpression_spec, make_soft_combo_spec,
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
    print(f"  {ct}: 1 young anchor, {n_old} old cells")

print(f"Total pairs: {len(pairs)}")

# %% — Build soft-OE specs
specs = [make_noop_spec()]
positive_single_ensgs = set()  # sOE{K}:{ENSG}
combo_spec_names = []           # sOE{K}_COMBO:{name}

for gene in SINGLE_TFS:
    try:
        eid = symbol_to_ensembl(gene)
    except KeyError as e:
        print(f"  SKIP: {e}")
        continue
    if eid not in token_dict:
        print(f"  SKIP: {gene} ({eid}) not in vocab")
        continue
    specs.append(make_soft_overexpression_spec(tokenizer, eid, boost_ranks=DEFAULT_BOOST_RANKS))
    positive_single_ensgs.add(eid)
    print(f"  sOE{DEFAULT_BOOST_RANKS} spec: {gene} -> {eid}")

for combo_name, gene_list in COMBO_TFS.items():
    try:
        eids = [symbol_to_ensembl(g) for g in gene_list]
    except KeyError as e:
        print(f"  SKIP combo {combo_name}: {e}")
        continue
    if not all(e in token_dict for e in eids):
        continue
    spec = make_soft_combo_spec(tokenizer, eids, boost_ranks=DEFAULT_BOOST_RANKS, name=combo_name)
    specs.append(spec)
    combo_spec_names.append(spec.name)
    print(f"  {spec.name}: {gene_list}")

# OSKM boost-rank sweep
try:
    oskm_eids = [symbol_to_ensembl(g) for g in COMBO_TFS["OSKM"]]
    oskm_sweep_names = []
    for k in OSKM_SWEEP_RANKS:
        if k == DEFAULT_BOOST_RANKS:
            continue  # already covered by the combo loop above
        spec = make_soft_combo_spec(tokenizer, oskm_eids, boost_ranks=k, name="OSKM")
        specs.append(spec)
        oskm_sweep_names.append(spec.name)
        print(f"  sweep: {spec.name}")
except KeyError as e:
    print(f"  SKIP OSKM sweep: {e}")
    oskm_sweep_names = []

# Null: 50 random single-gene soft OEs at K=DEFAULT_BOOST_RANKS
_rng = random.Random(SEED)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_null_pool = [e for e in _vocab_ensgs if e not in positive_single_ensgs]
null_ensgs = _rng.sample(_null_pool, N_NULL_GENES)
for eid in null_ensgs:
    specs.append(make_soft_overexpression_spec(tokenizer, eid, boost_ranks=DEFAULT_BOOST_RANKS))

print(
    f"\nSpecs: 1 baseline + {len(positive_single_ensgs)} single-TF sOE + "
    f"{len(combo_spec_names)} combo sOE + {len(oskm_sweep_names)} OSKM sweep + "
    f"{N_NULL_GENES} null = {len(specs)} total"
)

# %% — Build screen dataset
SCREEN_DIR = OUTPUT_DIR / "screen_soft_oe_v1"
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
print(f"Saved to {SCREEN_DIR / 'scored_results.csv'}")

# %% — Summary and z-scoring
null_spec_names = {f"sOE{DEFAULT_BOOST_RANKS}:{e}" for e in null_ensgs}
positive_single_spec_names = {f"sOE{DEFAULT_BOOST_RANKS}:{e}" for e in positive_single_ensgs}
positive_combo_spec_names = set(combo_spec_names) | set(oskm_sweep_names)

df["spec_class"] = "baseline"
df.loc[df["spec_name"].isin(null_spec_names), "spec_class"] = "null"
df.loc[df["spec_name"].isin(positive_single_spec_names), "spec_class"] = "positive_single"
df.loc[df["spec_name"].isin(positive_combo_spec_names), "spec_class"] = "positive_combo"

summary = (
    df[df["spec_name"] != "baseline"]
    .groupby(["spec_name", "cell_type"])
    .agg(
        mean_delta=("delta_vs_baseline", "mean"),
        std_delta=("delta_vs_baseline", "std"),
        n_cells=("delta_vs_baseline", "count"),
    )
    .sort_values("mean_delta", ascending=False)
)
print("\n=== Soft-OE Screen Results (top 30 positive, bottom 10) ===")
print(summary.head(30).to_string())
print("...")
print(summary.tail(10).to_string())
summary.to_csv(SCREEN_DIR / "summary_per_spec.csv")

null_per_gene = (
    df[df["spec_class"] == "null"]
    .groupby(["cell_type", "spec_name"])["delta_vs_baseline"]
    .mean()
)
null_stats = null_per_gene.groupby("cell_type").agg(["mean", "std", "count"])
print("\n=== Null soft-OE distribution per cell type ===")
print(null_stats.to_string())
null_stats.to_csv(SCREEN_DIR / "null_stats.csv")

positive_per_gene = (
    df[df["spec_class"].isin({"positive_single", "positive_combo"})]
    .groupby(["cell_type", "spec_name", "spec_class"])
    .agg(
        mean_delta=("delta_vs_baseline", "mean"),
        n_cells=("delta_vs_baseline", "count"),
    )
    .reset_index()
)
positive_per_gene["null_mean"] = positive_per_gene["cell_type"].map(null_stats["mean"])
positive_per_gene["null_std"] = positive_per_gene["cell_type"].map(null_stats["std"])
positive_per_gene["z_score"] = (
    (positive_per_gene["mean_delta"] - positive_per_gene["null_mean"]) / positive_per_gene["null_std"]
)
positive_per_gene = positive_per_gene.sort_values("z_score", ascending=False)
print("\n=== Positive TFs z-scored against null (|z|>2 is notable) ===")
print(positive_per_gene.to_string(index=False))
positive_per_gene.to_csv(SCREEN_DIR / "positive_tfs_zscored.csv", index=False)

# %% — OSKM boost-rank curve
osksweep = (
    df[df["spec_name"].str.endswith(":OSKM")]
    .groupby(["spec_name", "cell_type"])
    .agg(
        mean_delta=("delta_vs_baseline", "mean"),
        std_delta=("delta_vs_baseline", "std"),
        n_cells=("delta_vs_baseline", "count"),
    )
    .reset_index()
)
osksweep["boost_ranks"] = osksweep["spec_name"].str.extract(r"sOE(\d+)_COMBO").astype(int)
osksweep = osksweep.sort_values(["cell_type", "boost_ranks"])
print("\n=== OSKM boost-rank sweep (smaller K = more physiological) ===")
print(osksweep[["cell_type", "boost_ranks", "mean_delta", "std_delta", "n_cells"]].to_string(index=False))
osksweep.to_csv(SCREEN_DIR / "oskm_sweep.csv", index=False)

# %% [markdown]
# ## What to look for
#
# - **Soft OE at K=50 should pull the null much tighter.** In Phase 2 the
#   hepatocyte null std was 0.0096 (inflated by a few catastrophic pseudogene
#   outliers at rank 0); with K=50 those catastrophic picks should shrink.
# - **OSKM sweep** is the decisive plot: does `mean_delta(OSKM)` cross from
#   negative to positive as K shrinks from 2000 → 10? That would confirm the
#   Phase 2 negative signal was an OOD artifact, not a real "reprogramming
#   makes cells older" claim. If even at K=10 OSKM stays negative, MaxToki
#   genuinely doesn't model reprogramming biology at this scale.
# - **HNF4A in hepatocytes** should rise from z=-0.23 (Phase 2, hard OE) toward
#   positive if the soft operator is working — HNF4A is the hepatocyte master
#   regulator and should be in-distribution to boost mildly.
