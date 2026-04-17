# %% [markdown]
# # Phase 1: Tabula Sapiens Liver — Data Prep & Perturbation Screen
#
# Run inside the Apptainer container on a GPU node:
# ```bash
# srun -p gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
# module load apptainer
# apptainer shell --nv -B $HOME/maxToki:/workspace -B /ptmp/$USER:/ptmp/$USER $HOME/maxToki/maxtoki.sif
# cd /workspace
# python phase1_liver_screen.py          # or open in JupyterLab
# ```

# %% — Imports and paths
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)

H5AD_PATH = Path("/workspace/Liver_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad")
RESOURCES = Path("/workspace/resources")
TOKEN_DICT_PATH = RESOURCES / "token_dictionary_v1.json"
GENE_MEDIAN_PATH = RESOURCES / "gene_median_dictionary_v1.json"
ENSEMBL_MAP_PATH = RESOURCES / "ensembl_mapping_dict_v1.json"
HF_MODEL_PATH = Path("/ptmp/artfi/models/maxtoki-hf/MaxToki-217M-HF")
OUTPUT_DIR = Path("/ptmp/artfi/liver_screen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% — Load h5ad and inspect structure
adata = sc.read_h5ad(H5AD_PATH)
print(f"Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"\nobs columns: {list(adata.obs.columns)}")
print(f"\nvar columns: {list(adata.var.columns)}")
print(f"\nvar index (first 5): {list(adata.var.index[:5])}")

# %% — Explore cell types
if "cell_type" in adata.obs.columns:
    ct_col = "cell_type"
elif "cell_ontology_class" in adata.obs.columns:
    ct_col = "cell_ontology_class"
elif "cell_type_name" in adata.obs.columns:
    ct_col = "cell_type_name"
else:
    print("Available obs columns:", list(adata.obs.columns))
    ct_col = None  # MANUAL: set this to the correct column name after inspecting

if ct_col:
    print(f"\nUsing cell type column: '{ct_col}'")
    print(adata.obs[ct_col].value_counts())

# %% — Explore donor age
age_candidates = [c for c in adata.obs.columns if "age" in c.lower() or "donor" in c.lower()]
print(f"Age/donor columns: {age_candidates}")
for col in age_candidates:
    print(f"\n'{col}' unique values: {sorted(adata.obs[col].unique())[:20]}")

# %% [markdown]
# ## MANUAL CHECK
#
# After running the cells above, fill in these values based on what you see.
# Remove or comment out the `raise SystemExit(...)` below to continue past this point.

# ---- Config derived from exploration of TSP1_30 liver h5ad ----
# Donors: TSP6 (unknown age), TSP14, TSP17, TSP19 with ages [36, 59, 60, 67].
# Only age=36 is "young"; 59 is dropped as middle; 60 and 67 are "old".
CELL_TYPE_COL = "cell_ontology_class"
AGE_COL = "age"

FIBROSIS_CELL_TYPES = [
    "hepatocyte",
    "hepatic stellate cell",
    "intrahepatic cholangiocyte",
]

YOUNG_MAX = 36   # inclusive upper bound for "young" (only donor @ 36)
OLD_MIN = 60     # inclusive lower bound for "old"  (donors @ 60, 67)

# %% — Filter to fibrosis-relevant cell types
# Try case-insensitive matching
ct_values = adata.obs[CELL_TYPE_COL].str.lower()
target_lower = [t.lower() for t in FIBROSIS_CELL_TYPES]
mask_ct = ct_values.isin(target_lower)

if mask_ct.sum() == 0:
    # Try substring matching as fallback
    mask_ct = ct_values.apply(lambda x: any(t in x for t in target_lower))

print(f"Cells matching fibrosis cell types: {mask_ct.sum()} / {len(adata)}")
adata_fib = adata[mask_ct].copy()
print(adata_fib.obs[CELL_TYPE_COL].value_counts())

# %% — Parse ages and stratify young/old
# Ages may be strings like "30-35" or "40" or "40-44" — handle both
ages_raw = adata_fib.obs[AGE_COL]
print(f"Age dtype: {ages_raw.dtype}")
print(f"Sample values: {ages_raw.unique()[:10]}")


def parse_age(val):
    """Parse age value to a numeric midpoint. Handles '30', '30-35', '30s' formats."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().rstrip("s").rstrip("+")
    if "-" in s:
        parts = s.split("-")
        return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(s)
    except ValueError:
        return np.nan


adata_fib.obs["age_numeric"] = ages_raw.map(parse_age)
print(f"\nAge range: {adata_fib.obs['age_numeric'].min():.0f} – {adata_fib.obs['age_numeric'].max():.0f}")
print(f"Age distribution:\n{adata_fib.obs['age_numeric'].describe()}")

young_mask = adata_fib.obs["age_numeric"] <= YOUNG_MAX
old_mask = adata_fib.obs["age_numeric"] >= OLD_MIN

print(f"\nYoung cells (age <= {YOUNG_MAX}): {young_mask.sum()}")
print(f"Old cells (age >= {OLD_MIN}): {old_mask.sum()}")

# Per cell type breakdown
for ct in adata_fib.obs[CELL_TYPE_COL].unique():
    ct_mask = adata_fib.obs[CELL_TYPE_COL] == ct
    n_young = (ct_mask & young_mask).sum()
    n_old = (ct_mask & old_mask).sum()
    print(f"  {ct}: {n_young} young, {n_old} old")

adata_young = adata_fib[young_mask].copy()
adata_old = adata_fib[old_mask].copy()

# %% — Check required fields for MaxToki tokenization
print("Checking required fields...")

# 1. var.ensembl_id
has_ensembl_col = "ensembl_id" in adata_fib.var.columns
index_looks_ensembl = adata_fib.var.index[0].startswith("ENSG")
print(f"  var has 'ensembl_id' column: {has_ensembl_col}")
print(f"  var.index looks like Ensembl IDs: {index_looks_ensembl}")

if not has_ensembl_col and index_looks_ensembl:
    print("  -> Will use --use-h5ad-index flag for tokenization")
    USE_INDEX_AS_ENSEMBL = True
elif has_ensembl_col:
    USE_INDEX_AS_ENSEMBL = False
else:
    print("  WARNING: No Ensembl IDs found. Check var columns:")
    print(f"    var columns: {list(adata_fib.var.columns)}")
    print(f"    var index sample: {list(adata_fib.var.index[:5])}")
    USE_INDEX_AS_ENSEMBL = False

# 2. obs.n_counts
if "n_counts" not in adata_fib.obs.columns:
    print("  'n_counts' missing — computing from expression matrix...")
    adata_young.obs["n_counts"] = np.array(adata_young.X.sum(axis=1)).flatten()
    adata_old.obs["n_counts"] = np.array(adata_old.X.sum(axis=1)).flatten()
    print(f"  n_counts range (young): {adata_young.obs['n_counts'].min():.0f} – {adata_young.obs['n_counts'].max():.0f}")
else:
    print(f"  'n_counts' present, range: {adata_fib.obs['n_counts'].min():.0f} – {adata_fib.obs['n_counts'].max():.0f}")

# %% — Diagnose gene ID encoding and pick the right column for tokenization
# MaxToki's token dict is keyed by Ensembl IDs (ENSG...). The ensembl_mapping_dict
# is `gene_symbol -> ENSG` (upper-cased keys). We need var.ensembl_id to contain
# values that, after .str.upper().map(gene_mapping_dict), yield entries in the token dict.
with open(TOKEN_DICT_PATH) as f:
    _token_dict_probe = json.load(f)
with open(ENSEMBL_MAP_PATH) as f:
    _map_probe = json.load(f)

_probe = adata_fib.var
print(f"var.ensembl_id sample: {list(_probe['ensembl_id'].astype(str).iloc[:5])}")
print(f"var.gene_symbol sample: {list(_probe['gene_symbol'].astype(str).iloc[:5])}")
print(f"var.index sample: {list(_probe.index[:5])}")

_ens_hits_direct = _probe["ensembl_id"].astype(str).str.upper().isin(_token_dict_probe).sum()
_ens_hits_via_map = _probe["ensembl_id"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
_sym_hits_via_map = _probe["gene_symbol"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
_idx_hits_via_map = _probe.index.astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
print(f"  ensembl_id direct-hits in token_dict: {_ens_hits_direct}")
print(f"  ensembl_id -> mapping -> token_dict:  {_ens_hits_via_map}")
print(f"  gene_symbol -> mapping -> token_dict: {_sym_hits_via_map}")
print(f"  var.index  -> mapping -> token_dict:  {_idx_hits_via_map}")

# Pick whichever column has the most hits through the mapping dict.
_candidates = [
    ("ensembl_id", _ens_hits_via_map),
    ("gene_symbol", _sym_hits_via_map),
    ("index", _idx_hits_via_map),
]
_best_col, _best_hits = max(_candidates, key=lambda t: t[1])
print(f"Best column for tokenization: '{_best_col}' ({_best_hits} hits)")
if _best_hits < 1000:
    raise SystemExit(
        f"No column hits the vocab well enough ({_best_hits} hits). Inspect var columns manually."
    )

# %% — Save filtered h5ad files for tokenization
# The MaxToki tokenizer requires obs.time and obs.unique_cell_id columns, and
# --data-directory scans for ALL .h5ad files in the dir, so give each split
# its own input directory.
for adata_split in (adata_young, adata_old):
    adata_split.obs["time"] = 0  # Tabula Sapiens is cross-sectional; use 0 as placeholder
    adata_split.obs["unique_cell_id"] = adata_split.obs_names.astype(str)
    # time_group is defaulted (not None) in TranscriptomeTokenizer; group by donor
    # so each donor's cells form their own pseudo-trajectory group.
    adata_split.obs["time_group"] = adata_split.obs["donor"].astype(str)
    # Overwrite var.ensembl_id with whichever column actually maps to the vocab.
    if _best_col == "gene_symbol":
        adata_split.var["ensembl_id"] = adata_split.var["gene_symbol"].astype(str)
    elif _best_col == "index":
        adata_split.var["ensembl_id"] = adata_split.var.index.astype(str)

young_input_dir = OUTPUT_DIR / "input_young"
old_input_dir = OUTPUT_DIR / "input_old"
young_input_dir.mkdir(exist_ok=True)
old_input_dir.mkdir(exist_ok=True)
young_h5ad = young_input_dir / "ts_liver_young.h5ad"
old_h5ad = old_input_dir / "ts_liver_old.h5ad"
adata_young.write_h5ad(young_h5ad)
adata_old.write_h5ad(old_h5ad)
print(f"Saved {len(adata_young)} young cells -> {young_h5ad}")
print(f"Saved {len(adata_old)} old cells -> {old_h5ad}")

# %% — Tokenize with MaxToki pipeline
import subprocess

for label, input_dir in [("young", young_input_dir), ("old", old_input_dir)]:
    out_dir = OUTPUT_DIR / f"tokenized_{label}"
    out_dir.mkdir(exist_ok=True)

    cmd = [
        "python", "-m", "bionemo.maxtoki.data_prep", "tokenize",
        "--data-directory", str(input_dir),
        "--output-directory", str(out_dir),
        "--output-prefix", f"ts_liver_{label}",
        "--token-dictionary-file", str(TOKEN_DICT_PATH),
        "--gene-median-file", str(GENE_MEDIAN_PATH),
        "--gene-mapping-file", str(ENSEMBL_MAP_PATH),
        "--nproc", "4",
        "--model-input-size", "4096",
    ]
    if USE_INDEX_AS_ENSEMBL:
        cmd.append("--use-h5ad-index")

    print(f"\nTokenizing {label} cells...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:]}")
        raise SystemExit(f"Tokenization failed for {label}; see stderr above.")
    print(f"  Done: {out_dir}")

# %% — Load tokenized datasets and inspect
import datasets

ds_young_path = OUTPUT_DIR / "tokenized_young"
ds_old_path = OUTPUT_DIR / "tokenized_old"

# Find the .dataset directory inside the tokenized output
young_datasets = list(ds_young_path.glob("*.dataset"))
old_datasets = list(ds_old_path.glob("*.dataset"))
print(f"Young dataset dirs: {young_datasets}")
print(f"Old dataset dirs: {old_datasets}")

if young_datasets and old_datasets:
    ds_young = datasets.load_from_disk(str(young_datasets[0]))
    ds_old = datasets.load_from_disk(str(old_datasets[0]))
    print(f"\nYoung: {len(ds_young)} cells, sample length: {len(ds_young[0]['input_ids'])}")
    print(f"Old:   {len(ds_old)} cells, sample length: {len(ds_old[0]['input_ids'])}")
    print(f"Young sample first 10 tokens: {ds_young[0]['input_ids'][:10]}")
else:
    print("Tokenized datasets not found — check tokenization output above.")
    print("Listing output dirs:")
    for p in [ds_young_path, ds_old_path]:
        print(f"  {p}: {list(p.iterdir()) if p.exists() else 'MISSING'}")

# %% — Build CellPair objects for the screen
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
from bionemo.maxtoki.perturb import (
    CellPair, build_screen_dataset_scoring, score_screen,
    make_knockout_spec, make_noop_spec, make_overexpression_spec,
)

with open(TOKEN_DICT_PATH) as f:
    token_dict = json.load(f)
tokenizer = MaxTokiTokenizer(token_dict)

with open(ENSEMBL_MAP_PATH) as f:
    ensembl_map = json.load(f)


def symbol_to_ensembl(symbol):
    """Resolve a gene symbol to its Ensembl ID via the mapping dict."""
    eid = ensembl_map.get(symbol)
    if eid is None:
        raise KeyError(f"Gene symbol {symbol!r} not found in ensembl_mapping_dict_v1.json")
    return eid


# Limit to N cells per type for an initial run (full screen later)
MAX_CELLS_PER_TYPE = 50

# Pick representative young cells as anchors (one per cell type)
# and pair each old cell with a young anchor of the same type.
pairs = []
cell_types_in_data = adata_old.obs[CELL_TYPE_COL].unique()

for ct in cell_types_in_data:
    young_ct = ds_young.filter(
        lambda x, idx: adata_young.obs.iloc[idx][CELL_TYPE_COL] == ct,
        with_indices=True,
    ) if len(ds_young) > 0 else None
    old_ct = ds_old.filter(
        lambda x, idx: adata_old.obs.iloc[idx][CELL_TYPE_COL] == ct,
        with_indices=True,
    ) if len(ds_old) > 0 else None

    if young_ct is None or old_ct is None or len(young_ct) == 0 or len(old_ct) == 0:
        print(f"  Skipping {ct}: no young or old cells after tokenization")
        continue

    # Truncate both cells so (young + old + 4 special tokens) fits in 4096.
    # Each cell is kept to its top CELL_MAX_TOKENS by rank; higher-ranked genes
    # are more informative, so this is a mild information loss at worst.
    CELL_MAX_TOKENS = 2000  # 2*(2000+2) = 4004 < 4096

    def _truncate(tokens):
        # tokens come from the HF tokenizer dataset wrapped as <bos> ... <eos>.
        if len(tokens) <= CELL_MAX_TOKENS:
            return list(tokens)
        # keep the leading <bos>, the top-ranked genes, and the trailing <eos>
        bos = [tokens[0]] if tokens and tokens[0] == tokenizer.bos_id else []
        eos = [tokens[-1]] if tokens and tokens[-1] == tokenizer.eos_id else []
        body = tokens[len(bos): len(tokens) - len(eos)]
        body = body[: CELL_MAX_TOKENS - len(bos) - len(eos)]
        return [*bos, *body, *eos]

    # Use a single representative young anchor (median-length cell)
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

    print(f"  {ct}: 1 young anchor (len={len(anchor_tokens)}), {n_old} old cells -> {n_old} pairs")

print(f"\nTotal pairs: {len(pairs)}")

# %% — Define perturbation specs
# Positive controls: known anti-fibrotic targets whose KO should
# reduce predicted age affinity in hepatic stellate cells.
POSITIVE_CONTROL_GENES = [
    "TGFB1",    # master fibrosis driver
    "CCN2",     # downstream of TGFB1 (CTGF)
    "ACTA2",    # alpha-SMA, HSC activation marker
    "COL1A1",   # collagen, fibrosis hallmark
    "PDGFRA",   # HSC proliferation
    "LOX",      # collagen cross-linking
    "TIMP1",    # MMP inhibitor, ECM accumulation
]

specs = [make_noop_spec()]
positive_control_ensgs: set[str] = set()

for gene in POSITIVE_CONTROL_GENES:
    try:
        eid = symbol_to_ensembl(gene)
        if eid in token_dict:
            specs.append(make_knockout_spec(tokenizer, eid))
            positive_control_ensgs.add(eid)
            print(f"  KO spec: {gene} -> {eid} (token {token_dict[eid]})")
        else:
            print(f"  SKIP: {gene} -> {eid} not in MaxToki vocabulary")
    except KeyError as e:
        print(f"  SKIP: {e}")

# %% — Add null-distribution KOs for z-scoring
# Sample random Ensembl genes from the vocabulary; their mean_delta across cells
# defines the noise floor for each cell type. Effect rate of random genes will
# be low in most cell types (good: matches the sparsity of positive controls).
import random

N_NULL_GENES = 50  # 50 * 109 pairs ~= 5.5k forward passes, ~15-20 min on A100
_null_rng = random.Random(42)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_null_pool = [e for e in _vocab_ensgs if e not in positive_control_ensgs]
null_ensgs = _null_rng.sample(_null_pool, N_NULL_GENES)
for eid in null_ensgs:
    specs.append(make_knockout_spec(tokenizer, eid))

print(f"\nTotal specs: {len(specs)} "
      f"(1 baseline + {len(positive_control_ensgs)} positive + {N_NULL_GENES} null)")

# %% — Build screen dataset
SCREEN_DIR = OUTPUT_DIR / "screen_fibrosis_v1"
ds_path, manifest_path = build_screen_dataset_scoring(
    tokenizer,
    pairs,
    specs,
    output_dir=SCREEN_DIR,
    model_input_size=4096,
)
manifest = pd.read_csv(manifest_path)
print(f"Screen dataset: {len(manifest)} prompts")
print(f"  saved to: {ds_path}")
print(f"  manifest: {manifest_path}")
print(manifest["spec_name"].value_counts())

# %% — Score screen with MaxToki-217M-HF
import torch
from transformers import AutoModelForCausalLM

print("Loading MaxToki-217M-HF model...")
model = AutoModelForCausalLM.from_pretrained(
    str(HF_MODEL_PATH),
    torch_dtype=torch.bfloat16,
).to("cuda").eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

print("Scoring screen...")
df = score_screen(model, ds_path, manifest_path, device="cuda")
df.to_csv(SCREEN_DIR / "scored_results.csv", index=False)
print(f"Results saved to {SCREEN_DIR / 'scored_results.csv'}")

# %% — Diagnostic: how often did each KO actually change the prompt?
# If the KO gene isn't in the old cell's kept rank tokens, prompt_length matches
# the baseline exactly and delta_vs_baseline is identically 0.0.
baseline_lengths = (
    df[df["spec_name"] == "baseline"].set_index("pair_idx")["prompt_length"]
)
df["ko_took_effect"] = (
    df["pair_idx"].map(baseline_lengths) != df["prompt_length"]
)
effect_rates = (
    df[df["spec_name"] != "baseline"]
    .groupby(["spec_name", "cell_type"])["ko_took_effect"]
    .mean()
    .unstack()
)
print("\n=== Fraction of cells where KO removed at least one token ===")
print(effect_rates.to_string())

# %% — Analyze results
# Aggregate: mean delta_vs_baseline per (spec_name, cell_type)
if "delta_vs_baseline" in df.columns:
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
    print("\n=== Perturbation Screen Results ===")
    print("Positive delta = model thinks young state more likely = more rejuvenated\n")
    print(summary.to_string())

    # Highlight: do positive controls surface in HSCs?
    hsc_hits = summary.xs("hepatic stellate cell", level="cell_type", drop_level=False) if "hepatic stellate cell" in summary.index.get_level_values("cell_type") else pd.DataFrame()
    if not hsc_hits.empty:
        print("\n=== HSC-specific hits (expect TGFB1/CCN2 KO at top) ===")
        print(hsc_hits.sort_values("mean_delta", ascending=False).to_string())
else:
    print("No delta_vs_baseline column — check that baseline spec was included.")

# %% — Z-score positive controls against the null distribution
# For each cell type, the null is the per-gene mean_delta over the N_NULL_GENES
# random KOs. We restrict to cells where the KO actually took effect, because a
# zero-delta "no-op" KO artificially shrinks the null variance.
positive_spec_names = {f"KO:{e}" for e in positive_control_ensgs}
null_spec_names = {f"KO:{e}" for e in null_ensgs}

df["spec_class"] = "baseline"
df.loc[df["spec_name"].isin(positive_spec_names), "spec_class"] = "positive"
df.loc[df["spec_name"].isin(null_spec_names), "spec_class"] = "null"

effective = df[df["ko_took_effect"]]
null_per_gene = (
    effective[effective["spec_class"] == "null"]
    .groupby(["cell_type", "spec_name"])["delta_vs_baseline"]
    .mean()
)
null_stats = null_per_gene.groupby("cell_type").agg(["mean", "std", "count"])
print("\n=== Null-gene KO distribution per cell type (effective KOs only) ===")
print(null_stats.to_string())

positive_per_gene = (
    effective[effective["spec_class"] == "positive"]
    .groupby(["cell_type", "spec_name"])
    .agg(
        mean_delta=("delta_vs_baseline", "mean"),
        n_effective_cells=("delta_vs_baseline", "count"),
    )
)
positive_per_gene["null_mean"] = positive_per_gene.index.get_level_values("cell_type").map(null_stats["mean"])
positive_per_gene["null_std"] = positive_per_gene.index.get_level_values("cell_type").map(null_stats["std"])
positive_per_gene["z_score"] = (
    (positive_per_gene["mean_delta"] - positive_per_gene["null_mean"]) / positive_per_gene["null_std"]
)
print("\n=== Positive controls z-scored against null (|z|>2 is notable) ===")
print(positive_per_gene.sort_values("z_score", ascending=False).to_string())
positive_per_gene.to_csv(SCREEN_DIR / "positive_controls_zscored.csv")

# %% [markdown]
# ## Interpretation
#
# - **Positive `delta_vs_baseline`**: the perturbation makes the model assign
#   *higher* log-probability to a young successor state. The perturbed old cell
#   "looks younger" to the model.
# - **Negative delta**: perturbation makes the old cell look even older.
# - **Positive controls**: `KO:ENSG*` entries for TGFB1, CCN2, ACTA2 should
#   have the largest positive delta in HSCs if the model captures fibrosis
#   biology. If they don't surface, the pipeline needs debugging before
#   trusting novel hits.
# - **Next steps**: expand to ~500 senescence/fibrosis genes (SenMayo + KEGG
#   TGF-beta + druggable genome), then overlay with LINCS L1000 compound
#   signatures for drug repurposing.
