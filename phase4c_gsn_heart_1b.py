# %% [markdown]
# # Phase 4c: GSN replication on MaxToki-1B (capacity check)
#
# Phase 4b ruled out anchor-noise as the cause of the GSN null (10-anchor
# ensemble left z ≈ 0). The remaining variables are model capacity and readout
# head. This phase swaps the 217M checkpoint for 1B (same pretraining-only
# LL readout, same everything else) to isolate capacity.
#
# Changes vs Phase 4b:
#   - HF_MODEL_PATH → MaxToki-1B-HF (4× parameters)
#   - N_ANCHORS 10 → 5 (keep runtime sane: 1B forward is ~3× slower)
#   - Drop the sOE arm. Phase 4b showed the prompt-length-based effectiveness
#     filter is broken for sOE (only counts cells where the gene was absent
#     and sOE inserted it). Rather than ship the bug twice, skip sOE here;
#     a fixed version belongs in a separate phase if needed.
#
# Interpretation:
#   - If KO:GSN z moves from ≈ 0 to ≥ +2: capacity was the bottleneck — the
#     217M pretraining-only readout undercooks the age axis.
#   - If it stays null: capacity isn't the issue either. The remaining lever
#     is the fine-tuned TimeBetweenCells regression head, which isn't publicly
#     released — ending the replication line on this model.
#
# Reuses Phase 4 tokenization at /ptmp/$USER/heart_screen_phase4/tokenized_*.
# Outputs to /ptmp/$USER/heart_screen_phase4c/ and phase4c_results.md.

# %% — Imports and config
import json
import random
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
HF_MODEL_PATH = Path("/ptmp/artfi/models/maxtoki-hf/MaxToki-1B-HF")

PHASE4_DIR = Path("/ptmp/artfi/heart_screen_phase4")
OUTPUT_DIR = Path("/ptmp/artfi/heart_screen_phase4c")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_MAX_TOKENS = 2000
MAX_CELLS = 200
N_ANCHORS = 5
N_NULL_GENES = 50
SEED = 42
Z_NOTABLE = 2.0
TARGET_GENE = "GSN"

if not HF_MODEL_PATH.exists():
    raise SystemExit(
        f"MaxToki-1B-HF not found at {HF_MODEL_PATH}. Download it on the login node with:\n"
        "  huggingface-cli download theodoris-lab/MaxToki --include 'MaxToki-1B-HF/*' \\\n"
        f"    --local-dir {HF_MODEL_PATH.parent.parent}"
    )

# %% — Load tokenized datasets from Phase 4
ds_young_dir = next((PHASE4_DIR / "tokenized_young").glob("*.dataset"))
ds_old_dir = next((PHASE4_DIR / "tokenized_old").glob("*.dataset"))
ds_young = datasets.load_from_disk(str(ds_young_dir))
ds_old = datasets.load_from_disk(str(ds_old_dir))
print(f"Loaded: {len(ds_young)} young, {len(ds_old)} old (from Phase 4 tokenization)")

# %% — Tokenizer
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


# %% — Anchors + pairs (identical logic to Phase 4b but N_ANCHORS=5)
young_lengths = np.array([len(r["input_ids"]) for r in ds_young])
sorted_idx = np.argsort(young_lengths)
quantile_positions = np.linspace(0, len(sorted_idx) - 1, N_ANCHORS).astype(int)
anchor_idxs = sorted_idx[quantile_positions]
anchors = [_truncate(ds_young[int(i)]["input_ids"]) for i in anchor_idxs]
print(f"Picked {len(anchors)} anchors at idxs {anchor_idxs.tolist()}; "
      f"lengths {[len(a) for a in anchors]}")

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

# %% — Specs: baseline + KO:GSN + 50 null KO
specs = [make_noop_spec()]
gsn_eid = symbol_to_ensembl(TARGET_GENE)
if gsn_eid not in token_dict:
    raise SystemExit(f"{TARGET_GENE} ({gsn_eid}) not in MaxToki vocab")
specs.append(make_knockout_spec(tokenizer, gsn_eid))

_rng = random.Random(SEED)
_vocab_ensgs = [k for k in token_dict if isinstance(k, str) and k.startswith("ENSG")]
_null_pool = [e for e in _vocab_ensgs if e != gsn_eid]
null_ensgs = _rng.sample(_null_pool, N_NULL_GENES)
for eid in null_ensgs:
    specs.append(make_knockout_spec(tokenizer, eid))
print(f"Specs: 1 baseline + 1 target + {N_NULL_GENES} null = {len(specs)}")

# %% — Build screen dataset and score with 1B
SCREEN_DIR = OUTPUT_DIR / "screen_gsn_1b_v1"
ds_path, manifest_path = build_screen_dataset_scoring(
    tokenizer, pairs, specs, output_dir=SCREEN_DIR, model_input_size=4096,
)
manifest = pd.read_csv(manifest_path)
print(f"Screen: {len(manifest)} prompts -> {SCREEN_DIR}")

print(f"Loading MaxToki-1B-HF from {HF_MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    str(HF_MODEL_PATH), torch_dtype=torch.bfloat16,
).to("cuda").eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model: {n_params:.2f}B params")

print("Scoring...")
df = score_screen(model, ds_path, manifest_path, device="cuda")
df.to_csv(SCREEN_DIR / "scored_results.csv", index=False)

# %% — Ensemble and z-score
df["old_idx"] = df["cell_id"].str.extract(r"old(\d+)").astype(int)
df["anchor_idx"] = df["cell_id"].str.extract(r"anc(\d+)").astype(int)

baseline_lengths = (
    df[df["spec_name"] == "baseline"]
    .set_index(["old_idx", "anchor_idx"])["prompt_length"].to_dict()
)
df["_key"] = list(zip(df["old_idx"], df["anchor_idx"]))
df["baseline_len"] = df["_key"].map(baseline_lengths)
df["ko_took_effect"] = df["prompt_length"] != df["baseline_len"]
df = df.drop(columns=["_key", "baseline_len"])

eff = df[df["ko_took_effect"]]
per_old = (
    eff.groupby(["spec_name", "old_idx"])["delta_vs_baseline"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "delta_ens", "count": "n_anchors_eff"})
)
per_old = per_old[per_old["n_anchors_eff"] >= 2]  # N_ANCHORS=5 so require 2/5
print(f"Ensembled: {len(per_old)} (spec, old) rows with ≥2 effective anchors")

gsn_ko = f"KO:{gsn_eid}"
null_ko_specs = {f"KO:{e}" for e in null_ensgs}
per_old["spec_class"] = "other"
per_old.loc[per_old["spec_name"] == gsn_ko, "spec_class"] = "target"
per_old.loc[per_old["spec_name"].isin(null_ko_specs), "spec_class"] = "null"
per_old.to_csv(SCREEN_DIR / "per_old_ensemble.csv", index=False)

null_per_gene = (
    per_old[per_old["spec_class"] == "null"]
    .groupby("spec_name")["delta_ens"].mean()
)
null_mean, null_std = null_per_gene.mean(), null_per_gene.std()
n_null_genes = len(null_per_gene)

tgt = per_old[per_old["spec_class"] == "target"]
gsn_mean = tgt["delta_ens"].mean() if len(tgt) else float("nan")
gsn_std = tgt["delta_ens"].std() if len(tgt) else float("nan")
gsn_n = len(tgt)
gsn_z = (gsn_mean - null_mean) / null_std if null_std > 0 else float("nan")

print(
    f"\nGSN KO (1B, ensembled): mean={gsn_mean:+.4f}, n={gsn_n}, "
    f"null={null_mean:+.4f}±{null_std:.4f} (n_null={n_null_genes}), z={gsn_z:+.2f}"
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
md.append("# Phase 4c — GSN Cardiac-Fibroblast Replication: MaxToki-1B\n")
md.append(f"**Run:** {datetime.now().isoformat(timespec='seconds')}  ")
md.append("**Script:** `phase4c_gsn_heart_1b.py`  ")
md.append(f"**Model:** MaxToki-1B-HF ({n_params:.2f}B params, pretraining-only, LL readout)  ")
md.append("**Data:** TSP1_30 heart cardiac fibroblasts — reuses Phase 4 tokenization  ")
md.append(f"**Pairs:** {N_ANCHORS} anchors × {n_old} old = {len(pairs)}  ")
md.append(f"**Specs:** 1 baseline + 1 target + {N_NULL_GENES} null = {len(specs)}  ")
md.append(f"**Prompts scored:** {len(df)}\n")

md.append("## Summary\n")
md.append(df_to_md(pd.DataFrame([{
    "arm": "KO:GSN (1B)",
    "n_cells": gsn_n,
    "mean_delta_ens": gsn_mean,
    "null_mean": null_mean,
    "null_std": null_std,
    "n_null_genes": n_null_genes,
    "z_score": gsn_z,
}])))

md.append("## Capacity comparison\n")
md.append("| model | z(GSN KO) | n_cells | null_std |\n| --- | ---: | ---: | ---: |\n")
md.append("| MaxToki-217M (Phase 4, 1 anchor) | -0.83 | 490 | 0.0015 |\n")
md.append("| MaxToki-217M (Phase 4b, 10 anchors) | -0.31 | 198 | 0.0015 |\n")
md.append(f"| MaxToki-1B (Phase 4c, {N_ANCHORS} anchors) | {gsn_z:+.2f} | {gsn_n} | {null_std:.4f} |\n")

md.append("## Interpretation\n")
if np.isnan(gsn_z):
    md.append("1B KO:GSN has insufficient effective cells — inconclusive.\n")
elif gsn_z >= Z_NOTABLE:
    md.append(f"**GSN KO on 1B reaches z = {gsn_z:+.2f}.** Capacity was the bottleneck — "
              "217M undercooks the age axis. The paper's hit replicates with more scale.\n")
elif gsn_z > 0:
    md.append(f"**GSN KO on 1B = z = {gsn_z:+.2f}.** Direction matches but sub-threshold. "
              "Capacity helps marginally; the dominant bottleneck is elsewhere (readout head).\n")
else:
    md.append(f"**GSN KO on 1B = z = {gsn_z:+.2f}** — still null or negative. "
              "Capacity is not the bottleneck. The pretraining-only LL readout does not "
              "resolve GSN at any public scale of MaxToki. Remaining lever is the fine-tuned "
              "TimeBetweenCells head, which is not publicly released.\n")

md.append("\n## Provenance\n")
md.append(f"Outputs in `{SCREEN_DIR}/`:\n")
for f in ("scored_results.csv", "per_old_ensemble.csv"):
    md.append(f"- `{f}`")
md.append("\nRegenerate with `python phase4c_gsn_heart_1b.py`.\n")

md_path = OUTPUT_DIR / "phase4c_results.md"
md_path.write_text("\n".join(md))
print(f"Wrote {md_path}")

repo_md = Path("/workspace/phase4c_results.md")
try:
    repo_md.write_text("\n".join(md))
    print(f"Wrote {repo_md}")
except OSError as e:
    print(f"Could not write repo copy: {e}")

print("Done.")
