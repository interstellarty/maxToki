# %% [markdown]
# # Phase 4: GSN KO in Cardiac Fibroblasts (TSP heart) — replication attempt
#
# The MaxToki manuscript reports GSN as a rejuvenating hit in cardiac
# fibroblasts. This script tests whether the pretraining-only 217M checkpoint
# reproduces that signal on TSP1_30 heart:
#
#   - Load TSP heart h5ad, filter to cardiac fibroblasts only.
#   - Split young (≤ YOUNG_MAX_AGE) vs old (≥ OLD_MIN_AGE) donors.
#   - Tokenize each split with the standard MaxToki data-prep pipeline.
#   - Pair each old cell with one median-length young anchor, identical to
#     Phase 3.
#   - Score: baseline (no-op), KO:GSN, KO:50 random null genes.
#   - Restrict to cells where GSN was actually in the top-2000 tokens
#     (`ko_took_effect`). Z-score GSN mean-delta against the null.
#
# Interpretation: `delta_vs_baseline > 0` means the perturbed old cell now looks
# more like the young anchor (rejuvenation). GSN is pro-fibrotic in cardiac
# literature, so KO is the directional expectation. z ≥ +2.0 counts as a hit.
#
# Outputs:
#   - `/ptmp/$USER/heart_screen_phase4/` (tokenized data + CSVs)
#   - `phase4_results.md` at the repo root

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

H5AD_PATH = Path("/workspace/Heart_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad")
RESOURCES = Path("/workspace/resources")
TOKEN_DICT_PATH = RESOURCES / "token_dictionary_v1.json"
GENE_MEDIAN_PATH = RESOURCES / "gene_median_dictionary_v1.json"
ENSEMBL_MAP_PATH = RESOURCES / "ensembl_mapping_dict_v1.json"
HF_MODEL_PATH = Path("/ptmp/artfi/models/maxtoki-hf/MaxToki-217M-HF")
OUTPUT_DIR = Path("/ptmp/artfi/heart_screen_phase4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_TYPE_COL = "cell_ontology_class"
AGE_COL = "age"
FIBROBLAST_SUBSTR = "fibroblast"   # substring-match in cell_ontology_class (case-insensitive)
YOUNG_MAX_AGE = 40
OLD_MIN_AGE = 55

CELL_MAX_TOKENS = 2000
MAX_CELLS = 500
N_NULL_GENES = 50
SEED = 42
Z_NOTABLE = 2.0
TARGET_GENE = "GSN"

# %% — Load h5ad and inspect
adata = sc.read_h5ad(H5AD_PATH)
print(f"Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"obs columns: {list(adata.obs.columns)[:15]}...")

# Pick the cell type column that exists.
for cand in ("cell_ontology_class", "cell_type", "cell_type_name"):
    if cand in adata.obs.columns:
        CELL_TYPE_COL = cand
        break
print(f"Using cell type column: {CELL_TYPE_COL!r}")

# %% — Filter to cardiac fibroblasts
ct_lower = adata.obs[CELL_TYPE_COL].astype(str).str.lower()
fib_mask = ct_lower.str.contains(FIBROBLAST_SUBSTR)
print(f"Fibroblast-matching cells: {fib_mask.sum()} / {len(adata)}")
print("Matching cell types:")
print(adata.obs.loc[fib_mask, CELL_TYPE_COL].value_counts())

adata_fib = adata[fib_mask].copy()
if len(adata_fib) == 0:
    raise SystemExit("No cardiac fibroblasts found — inspect cell type labels above.")

# %% — Age parsing and young/old split

def parse_age(val):
    """Parse '30' / '30-35' / '30s' to a numeric midpoint."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().rstrip("s").rstrip("+")
    if "-" in s:
        a, b = s.split("-")
        return (float(a) + float(b)) / 2
    try:
        return float(s)
    except ValueError:
        return np.nan


adata_fib.obs["age_numeric"] = adata_fib.obs[AGE_COL].map(parse_age)
print(
    f"Age range: {adata_fib.obs['age_numeric'].min():.0f} – "
    f"{adata_fib.obs['age_numeric'].max():.0f}"
)
print(adata_fib.obs["age_numeric"].value_counts().sort_index())

young_mask = adata_fib.obs["age_numeric"] <= YOUNG_MAX_AGE
old_mask = adata_fib.obs["age_numeric"] >= OLD_MIN_AGE
print(f"\nYoung cells (age ≤ {YOUNG_MAX_AGE}): {young_mask.sum()}")
print(f"Old cells (age ≥ {OLD_MIN_AGE}): {old_mask.sum()}")
if young_mask.sum() == 0 or old_mask.sum() == 0:
    raise SystemExit("Empty young or old split — adjust YOUNG_MAX_AGE / OLD_MIN_AGE.")

adata_young = adata_fib[young_mask].copy()
adata_old = adata_fib[old_mask].copy()

# %% — Prepare h5ads for tokenization (same conventions as Phase 1)
# MaxToki tokenizer needs obs.time, obs.unique_cell_id, obs.time_group and
# var.ensembl_id columns that map into the token dict.
with open(TOKEN_DICT_PATH) as f:
    _token_dict_probe = json.load(f)
with open(ENSEMBL_MAP_PATH) as f:
    _map_probe = json.load(f)

_probe = adata_fib.var
has_ensembl_col = "ensembl_id" in _probe.columns
has_symbol_col = "gene_symbol" in _probe.columns
idx_looks_ensembl = str(_probe.index[0]).startswith("ENSG")

_hits = {}
if has_ensembl_col:
    _hits["ensembl_id"] = _probe["ensembl_id"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
if has_symbol_col:
    _hits["gene_symbol"] = _probe["gene_symbol"].astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()
if idx_looks_ensembl:
    _hits["index"] = _probe.index.astype(str).str.upper().map(_map_probe).isin(_token_dict_probe).sum()

print(f"Token-dict hits per candidate var column: {_hits}")
_best_col = max(_hits, key=_hits.get)
print(f"Best column for tokenization: {_best_col!r} ({_hits[_best_col]} hits)")
if _hits[_best_col] < 1000:
    raise SystemExit(f"No column hits the vocab sufficiently ({_hits}).")

if "n_counts" not in adata_fib.obs.columns:
    adata_young.obs["n_counts"] = np.array(adata_young.X.sum(axis=1)).flatten()
    adata_old.obs["n_counts"] = np.array(adata_old.X.sum(axis=1)).flatten()

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
young_h5ad = young_input_dir / "ts_heart_young.h5ad"
old_h5ad = old_input_dir / "ts_heart_old.h5ad"
adata_young.write_h5ad(young_h5ad)
adata_old.write_h5ad(old_h5ad)
print(f"Wrote {len(adata_young)} young, {len(adata_old)} old cells.")

# %% — Tokenize both splits
USE_INDEX_AS_ENSEMBL = _best_col == "index" or (_best_col == "ensembl_id" and not has_ensembl_col)

for label, input_dir in [("young", young_input_dir), ("old", old_input_dir)]:
    out_dir = OUTPUT_DIR / f"tokenized_{label}"
    if any(out_dir.glob("*.dataset")):
        print(f"Tokenized {label} dataset already exists at {out_dir} — skipping.")
        continue
    out_dir.mkdir(exist_ok=True)
    cmd = [
        "python", "-m", "bionemo.maxtoki.data_prep", "tokenize",
        "--data-directory", str(input_dir),
        "--output-directory", str(out_dir),
        "--output-prefix", f"ts_heart_{label}",
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

ds_young = datasets.load_from_disk(str(next((OUTPUT_DIR / "tokenized_young").glob("*.dataset"))))
ds_old = datasets.load_from_disk(str(next((OUTPUT_DIR / "tokenized_old").glob("*.dataset"))))
print(f"Tokenized: {len(ds_young)} young, {len(ds_old)} old")

# %% — Build CellPairs (single cell type: cardiac fibroblast)
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


young_lengths = [len(r["input_ids"]) for r in ds_young]
anchor_idx = int(np.argsort(young_lengths)[len(young_lengths) // 2])
anchor_tokens = _truncate(ds_young[anchor_idx]["input_ids"])
print(f"Young anchor: idx={anchor_idx}, len={len(anchor_tokens)}")

n_old = min(len(ds_old), MAX_CELLS)
pairs = []
for i in range(n_old):
    pairs.append(CellPair(
        young_tokens=anchor_tokens,
        old_tokens=_truncate(ds_old[i]["input_ids"]),
        cell_id=f"cardiac_fibroblast:{i}",
        metadata={"cell_type": "cardiac_fibroblast"},
    ))
print(f"Built {len(pairs)} pairs (1 anchor × {n_old} old cells).")

# %% — Specs: baseline + KO:GSN + 50 null KOs
specs = [make_noop_spec()]
skipped = []
try:
    gsn_eid = symbol_to_ensembl(TARGET_GENE)
    if gsn_eid not in token_dict:
        raise KeyError(f"{TARGET_GENE} Ensembl {gsn_eid} not in MaxToki vocab")
    specs.append(make_knockout_spec(tokenizer, gsn_eid))
    print(f"KO spec: {TARGET_GENE} -> {gsn_eid}")
except KeyError as e:
    raise SystemExit(f"Cannot build GSN KO spec: {e}")

_rng = random.Random(SEED)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_null_pool = [e for e in _vocab_ensgs if e != gsn_eid]
null_ensgs = _rng.sample(_null_pool, N_NULL_GENES)
for eid in null_ensgs:
    specs.append(make_knockout_spec(tokenizer, eid))
print(f"Total specs: {len(specs)} (1 baseline + 1 GSN + {N_NULL_GENES} null)")

# %% — Build screen dataset and score
SCREEN_DIR = OUTPUT_DIR / "screen_gsn_v1"
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

# %% — Effect rates + z-score
baseline_lengths = df[df["spec_name"] == "baseline"].set_index("pair_idx")["prompt_length"]
df["ko_took_effect"] = df["pair_idx"].map(baseline_lengths) != df["prompt_length"]

gsn_spec = f"KO:{gsn_eid}"
null_specs = {f"KO:{e}" for e in null_ensgs}

df["spec_class"] = "baseline"
df.loc[df["spec_name"] == gsn_spec, "spec_class"] = "target"
df.loc[df["spec_name"].isin(null_specs), "spec_class"] = "null"

effective = df[df["ko_took_effect"]]

null_per_gene = (
    effective[effective["spec_class"] == "null"]
    .groupby("spec_name")["delta_vs_baseline"].mean()
)
null_mean = null_per_gene.mean()
null_std = null_per_gene.std()
n_effective_null_genes = len(null_per_gene)
print(f"Null: mean={null_mean:+.4f}, std={null_std:.4f} over {n_effective_null_genes} effective genes")

gsn_effective = effective[effective["spec_class"] == "target"]
n_gsn_effective = len(gsn_effective)
gsn_mean = gsn_effective["delta_vs_baseline"].mean() if n_gsn_effective else float("nan")
gsn_std = gsn_effective["delta_vs_baseline"].std() if n_gsn_effective else float("nan")
gsn_z = (gsn_mean - null_mean) / null_std if null_std > 0 else float("nan")
gsn_effect_rate = df[df["spec_name"] == gsn_spec]["ko_took_effect"].mean()

print(
    f"GSN: n_effective={n_gsn_effective} ({gsn_effect_rate:.2%}), "
    f"mean_delta={gsn_mean:+.4f}, z={gsn_z:+.2f}"
)

# Save per-gene null aggregates for the report.
null_per_gene.to_csv(SCREEN_DIR / "null_per_gene.csv", header=["mean_delta"])

# %% — Markdown report
def df_to_md(df_: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    if df_.empty:
        return "_(no rows)_\n"
    cols = list(df_.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df_.iterrows():
        cells = [fmt.format(v) if isinstance(v, float) else str(v) for v in (r[c] for c in cols)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


md = []
md.append("# Phase 4 — GSN KO in Cardiac Fibroblasts: Results\n")
md.append(f"**Run:** {datetime.now().isoformat(timespec='seconds')}  ")
md.append("**Script:** `phase4_gsn_heart.py`  ")
md.append("**Model:** MaxToki-217M-HF (pretraining-only, log-likelihood readout)  ")
md.append(f"**Data:** TSP1_30 heart — cardiac fibroblasts, {len(adata_young)} young / {len(adata_old)} old  ")
md.append(f"**Age split:** young ≤ {YOUNG_MAX_AGE}, old ≥ {OLD_MIN_AGE}  ")
md.append(f"**Pairs:** 1 young anchor × {len(pairs)} old cells = {len(pairs)} pairs  ")
md.append(f"**Specs:** 1 baseline + 1 GSN KO + {N_NULL_GENES} null KOs = {len(specs)}\n")

md.append("## Summary\n")
md.append(df_to_md(pd.DataFrame([{
    "gene": TARGET_GENE,
    "effect_rate": gsn_effect_rate,
    "n_effective": n_gsn_effective,
    "mean_delta": gsn_mean,
    "std_delta": gsn_std,
    "null_mean": null_mean,
    "null_std": null_std,
    "z_score": gsn_z,
}])))

md.append("## Interpretation\n")
if np.isnan(gsn_z):
    md.append("GSN KO had no effective cells — GSN absent from top-2000 tokens of every old "
              "cardiac fibroblast. Replication **inconclusive** (expression bottleneck).\n")
elif gsn_z >= Z_NOTABLE:
    md.append(f"**GSN KO reaches z = {gsn_z:+.2f} (≥ +{Z_NOTABLE}).** Directionally consistent "
              "with the MaxToki paper: KO of GSN makes the old cardiac fibroblast's transcriptome "
              "look more like its young counterpart. Replication **supported**.\n")
elif gsn_z > 0:
    md.append(f"**GSN KO z = {gsn_z:+.2f}** — direction matches the published hit but effect "
              f"is sub-threshold (|z| < {Z_NOTABLE}). Replication **weak/partial**.\n")
else:
    md.append(f"**GSN KO z = {gsn_z:+.2f}** — opposite direction from the published hit. "
              "Either (a) the paper used the fine-tuned TimeBetweenCells head and pretraining "
              "LL does not capture the signal, (b) anchor choice or age split is too different, "
              "or (c) the effect is fragile. Replication **not supported**.\n")

md.append(f"\n## Null distribution ({N_NULL_GENES} random-gene KOs, effective only)\n")
md.append(f"- Effective null genes: {n_effective_null_genes}\n")
md.append(f"- Null mean delta: {null_mean:+.4f}\n")
md.append(f"- Null std: {null_std:.4f}\n")

md.append("## Provenance\n")
md.append(f"Outputs in `{SCREEN_DIR}/`:\n")
for f in ("scored_results.csv", "null_per_gene.csv"):
    md.append(f"- `{f}`")
md.append("\nRegenerate with `python phase4_gsn_heart.py`.\n")

md_path = OUTPUT_DIR / "phase4_results.md"
md_path.write_text("\n".join(md))
print(f"Wrote {md_path}")

repo_md = Path("/workspace/phase4_results.md")
try:
    repo_md.write_text("\n".join(md))
    print(f"Wrote {repo_md}")
except OSError as e:
    print(f"Could not write repo copy: {e}")

print("Done.")
