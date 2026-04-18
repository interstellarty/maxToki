# %% [markdown]
# # Phase 2: Transcription-Factor Overexpression Screen — Can Boosting TFs Rejuvenate Old Liver Cells?
#
# Same TSP liver cohort and cell-type selection as Phase 1, but instead of
# knocking out fibrosis genes we **overexpress** (promote to rank 0) candidate
# rejuvenation transcription factors and ask whether the model now assigns a
# higher log-probability to a young-cell continuation.
#
# Readout: `delta_vs_baseline = mean_logprob(OE_old -> young) - mean_logprob(old -> young)`.
# Positive delta = model thinks young successor more likely = more "rejuvenated".
#
# Run inside the Apptainer container on a GPU node:
# ```bash
# srun -p gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
# module load apptainer
# apptainer shell --nv -B $HOME/maxToki:/workspace -B /ptmp/$USER:/ptmp/$USER $HOME/maxToki/maxtoki.sif
# cd /workspace
# python phase2_tf_rejuvenation.py
# ```
#
# If Phase 1 has already run, the tokenized datasets at
# `/ptmp/artfi/liver_screen/tokenized_{young,old}` are reused and we skip
# straight to screen construction.

# %% — Imports and paths
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

H5AD_PATH = Path("/workspace/Liver_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad")
RESOURCES = Path("/workspace/resources")
TOKEN_DICT_PATH = RESOURCES / "token_dictionary_v1.json"
GENE_MEDIAN_PATH = RESOURCES / "gene_median_dictionary_v1.json"
ENSEMBL_MAP_PATH = RESOURCES / "ensembl_mapping_dict_v1.json"
HF_MODEL_PATH = Path("/ptmp/artfi/models/maxtoki-hf/MaxToki-217M-HF")

# Reuse Phase 1's output dir so the tokenized datasets can be shared.
PHASE1_DIR = Path("/ptmp/artfi/liver_screen")
OUTPUT_DIR = Path("/ptmp/artfi/liver_screen_phase2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_TYPE_COL = "cell_ontology_class"
AGE_COL = "age"
FIBROSIS_CELL_TYPES = [
    "hepatocyte",
    "hepatic stellate cell",
    "intrahepatic cholangiocyte",
]
YOUNG_MAX = 36
OLD_MIN = 60
CELL_MAX_TOKENS = 2000  # 2*(2000+2) = 4004 < 4096 model_input_size
MAX_CELLS_PER_TYPE = 50
N_NULL_GENES = 50  # random single-gene OEs to define the null per cell type
SEED = 42

# %% — Candidate rejuvenation TFs
# Single-gene OEs: classic reprogramming + stemness + longevity + liver identity.
# Combo OEs: the canonical Yamanaka cocktails.
REJUV_TFS_REPROGRAMMING = ["POU5F1", "SOX2", "KLF4", "MYC", "NANOG", "LIN28A", "LIN28B"]
REJUV_TFS_LONGEVITY = ["FOXO3", "TET1", "TET2", "SIRT1", "SIRT6"]
REJUV_TFS_LIVER_IDENTITY = ["HNF4A", "HNF1A", "FOXA2", "CEBPA"]
SINGLE_TFS = REJUV_TFS_REPROGRAMMING + REJUV_TFS_LONGEVITY + REJUV_TFS_LIVER_IDENTITY

COMBO_TFS = {
    "OSK":    ["POU5F1", "SOX2", "KLF4"],            # partial reprogramming, Ocampo 2016
    "OSKM":   ["POU5F1", "SOX2", "KLF4", "MYC"],     # full Yamanaka
    "OSK_L":  ["POU5F1", "SOX2", "KLF4", "LIN28A"],  # stemness + reprogramming
    "NANOG_L": ["NANOG", "LIN28A"],                   # pluripotency maintenance
}

# %% — Load or create tokenized young/old datasets
import datasets

ds_young_root = PHASE1_DIR / "tokenized_young"
ds_old_root = PHASE1_DIR / "tokenized_old"
adata_young_path = PHASE1_DIR / "input_young" / "ts_liver_young.h5ad"
adata_old_path = PHASE1_DIR / "input_old" / "ts_liver_old.h5ad"

phase1_ready = (
    any(ds_young_root.glob("*.dataset")) if ds_young_root.exists() else False
) and (
    any(ds_old_root.glob("*.dataset")) if ds_old_root.exists() else False
) and adata_young_path.exists() and adata_old_path.exists()

if phase1_ready:
    print("Reusing Phase 1 tokenized datasets.")
    import anndata as ad
    adata_young = ad.read_h5ad(adata_young_path)
    adata_old = ad.read_h5ad(adata_old_path)
else:
    print("Phase 1 outputs not found — re-running data prep from scratch.")
    import scanpy as sc
    import subprocess

    adata = sc.read_h5ad(H5AD_PATH)
    print(f"Loaded {adata.shape[0]} cells x {adata.shape[1]} genes")

    ct_values = adata.obs[CELL_TYPE_COL].str.lower()
    target_lower = [t.lower() for t in FIBROSIS_CELL_TYPES]
    mask_ct = ct_values.isin(target_lower)
    if mask_ct.sum() == 0:
        mask_ct = ct_values.apply(lambda x: any(t in x for t in target_lower))
    adata_fib = adata[mask_ct].copy()

    def parse_age(val):
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

    adata_fib.obs["age_numeric"] = adata_fib.obs[AGE_COL].map(parse_age)
    young_mask = adata_fib.obs["age_numeric"] <= YOUNG_MAX
    old_mask = adata_fib.obs["age_numeric"] >= OLD_MIN
    adata_young = adata_fib[young_mask].copy()
    adata_old = adata_fib[old_mask].copy()
    print(f"Young: {len(adata_young)} cells, Old: {len(adata_old)} cells")

    # Tokenizer requirements (see data_prep/transcriptome_tokenizer.py).
    if "n_counts" not in adata_young.obs.columns:
        adata_young.obs["n_counts"] = np.array(adata_young.X.sum(axis=1)).flatten()
        adata_old.obs["n_counts"] = np.array(adata_old.X.sum(axis=1)).flatten()

    # Pick the var column whose IDs map best into the MaxToki vocab.
    with open(TOKEN_DICT_PATH) as f:
        _token_dict_probe = json.load(f)
    with open(ENSEMBL_MAP_PATH) as f:
        _map_probe = json.load(f)

    _probe = adata_fib.var
    _hits = {
        "ensembl_id": _probe["ensembl_id"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
                      if "ensembl_id" in _probe.columns else 0,
        "gene_symbol": _probe["gene_symbol"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
                      if "gene_symbol" in _probe.columns else 0,
        "index": _probe.index.astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum(),
    }
    _best_col = max(_hits, key=_hits.get)
    print(f"Using var column '{_best_col}' ({_hits[_best_col]} vocab hits) for tokenization.")
    if _hits[_best_col] < 1000:
        raise SystemExit("No var column hits the vocab well enough; inspect h5ad manually.")

    for adata_split in (adata_young, adata_old):
        adata_split.obs["time"] = 0
        adata_split.obs["unique_cell_id"] = adata_split.obs_names.astype(str)
        adata_split.obs["time_group"] = adata_split.obs["donor"].astype(str)
        if _best_col == "gene_symbol":
            adata_split.var["ensembl_id"] = adata_split.var["gene_symbol"].astype(str)
        elif _best_col == "index":
            adata_split.var["ensembl_id"] = adata_split.var.index.astype(str)

    young_input_dir = OUTPUT_DIR / "input_young"
    old_input_dir = OUTPUT_DIR / "input_old"
    young_input_dir.mkdir(exist_ok=True)
    old_input_dir.mkdir(exist_ok=True)
    adata_young.write_h5ad(young_input_dir / "ts_liver_young.h5ad")
    adata_old.write_h5ad(old_input_dir / "ts_liver_old.h5ad")

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
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            raise SystemExit(f"Tokenization failed for {label}")
    ds_young_root = OUTPUT_DIR / "tokenized_young"
    ds_old_root = OUTPUT_DIR / "tokenized_old"

ds_young = datasets.load_from_disk(str(next(ds_young_root.glob("*.dataset"))))
ds_old = datasets.load_from_disk(str(next(ds_old_root.glob("*.dataset"))))
print(f"Tokenized: {len(ds_young)} young / {len(ds_old)} old cells.")

# %% — Build CellPairs
from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
from bionemo.maxtoki.perturb import (
    CellPair, build_screen_dataset_scoring, score_screen,
    make_noop_spec, make_overexpression_spec, make_combo_spec,
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
        print(f"  Skipping {ct}: no young or old cells")
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
    print(f"  {ct}: 1 young anchor (len={len(anchor_tokens)}), {n_old} old cells")

print(f"Total pairs: {len(pairs)}")

# %% — Build OE specs: positives + combos + null
specs = [make_noop_spec()]
positive_oe_ensgs = set()
skipped_symbols = []

for gene in SINGLE_TFS:
    try:
        eid = symbol_to_ensembl(gene)
        if eid in token_dict:
            specs.append(make_overexpression_spec(tokenizer, eid))
            positive_oe_ensgs.add(eid)
            print(f"  OE spec: {gene} -> {eid} (token {token_dict[eid]})")
        else:
            skipped_symbols.append(gene)
            print(f"  SKIP: {gene} -> {eid} not in MaxToki vocabulary")
    except KeyError as e:
        skipped_symbols.append(gene)
        print(f"  SKIP: {e}")

combo_specs_built = []
for combo_name, gene_list in COMBO_TFS.items():
    try:
        eids = [symbol_to_ensembl(g) for g in gene_list]
    except KeyError as e:
        print(f"  SKIP combo {combo_name}: {e}")
        continue
    if not all(e in token_dict for e in eids):
        missing = [g for g, e in zip(gene_list, eids) if e not in token_dict]
        print(f"  SKIP combo {combo_name}: {missing} not in vocab")
        continue
    spec = make_combo_spec(tokenizer, overexpressions=eids, name=f"OE_COMBO:{combo_name}")
    specs.append(spec)
    combo_specs_built.append(combo_name)
    print(f"  OE combo spec: {combo_name} = {gene_list}")

# Null: random single-gene OEs from the rest of the vocab.
_rng = random.Random(SEED)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_null_pool = [e for e in _vocab_ensgs if e not in positive_oe_ensgs]
null_ensgs = _rng.sample(_null_pool, N_NULL_GENES)
for eid in null_ensgs:
    specs.append(make_overexpression_spec(tokenizer, eid))

print(
    f"\nSpecs: 1 baseline + {len(positive_oe_ensgs)} single-TF OE + "
    f"{len(combo_specs_built)} combo OE + {N_NULL_GENES} null OE = {len(specs)} total"
)

# %% — Build screen dataset
SCREEN_DIR = OUTPUT_DIR / "screen_rejuv_tfs_v1"
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
print(f"Saved scored results to {SCREEN_DIR / 'scored_results.csv'}")

# %% — Summarize per (spec, cell_type)
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
print("\n=== OE Screen Results (positive delta = more rejuvenated) ===")
print(summary.to_string())
summary.to_csv(SCREEN_DIR / "summary_per_spec.csv")

# %% — Z-score positives and combos against null
# Unlike KO, OE always changes the prompt (gene is inserted at rank 0 even when
# absent), so every row contributes to the null — no ko_took_effect filter.
null_spec_names = {f"OE:{e}" for e in null_ensgs}
positive_spec_names = {f"OE:{e}" for e in positive_oe_ensgs}
combo_spec_names = {f"OE_COMBO:{n}" for n in combo_specs_built}

df["spec_class"] = "baseline"
df.loc[df["spec_name"].isin(null_spec_names), "spec_class"] = "null"
df.loc[df["spec_name"].isin(positive_spec_names), "spec_class"] = "positive_single"
df.loc[df["spec_name"].isin(combo_spec_names), "spec_class"] = "positive_combo"

null_per_gene = (
    df[df["spec_class"] == "null"]
    .groupby(["cell_type", "spec_name"])["delta_vs_baseline"]
    .mean()
)
null_stats = null_per_gene.groupby("cell_type").agg(["mean", "std", "count"])
print("\n=== Null OE distribution per cell type ===")
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
print("\n=== Rejuvenation TFs z-scored against null (|z|>2 is notable) ===")
print(positive_per_gene.to_string(index=False))
positive_per_gene.to_csv(SCREEN_DIR / "positive_tfs_zscored.csv", index=False)

# %% [markdown]
# ## Interpretation
#
# - **Positive `delta_vs_baseline`** → overexpressing this TF in the old cell
#   makes the model assign higher log-probability to a young-cell continuation.
#   The model treats the perturbed old cell as closer to the young state.
# - **Negative delta** → OE of this TF looks *less* young-like (expected for
#   random genes that just disrupt rank structure).
# - **Top hits to watch** in HSCs + hepatocytes + cholangiocytes:
#     - OSK / OSKM combos (Ocampo 2016 partial reprogramming)
#     - HNF4A (hepatocyte master identity; age-associated decline)
#     - FOXO3, SIRT1, SIRT6 (longevity regulators)
# - **Caveats**: MaxToki was trained on normal single-cell transcriptomes,
#   not on reprogramming trajectories. A strong signal here means the young
#   rank-value signature of the target cell type contains the TF near the top;
#   a null signal doesn't rule out a real biological effect.
