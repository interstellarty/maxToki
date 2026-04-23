# Assemble tissue.h5ad from the MTX + metadata files produced by the companion
# R script (convert_ramachandran_to_h5ad.R). Run once, after the R step.
#
# Usage (inside the apptainer container, where scanpy/anndata are available):
#   module load apptainer
#   apptainer exec -B $HOME/maxToki:/workspace -B /ptmp/$USER:/ptmp/$USER \
#       $HOME/maxToki/maxtoki.sif \
#       bash -c "cd /ptmp/$USER/liver_fibrosis_phase5 && \
#                python /workspace/assemble_ramachandran_h5ad.py"

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio

HERE = Path.cwd()
counts_path = HERE / "counts.mtx"
genes_path = HERE / "genes.tsv"
barcodes_path = HERE / "barcodes.tsv"
obs_path = HERE / "obs.csv"
out_path = HERE / "tissue.h5ad"

for p in (counts_path, genes_path, barcodes_path, obs_path):
    if not p.exists():
        raise SystemExit(
            f"Missing {p}. Run convert_ramachandran_to_h5ad.R in this directory first."
        )

# Matrix Market as written by R writeMM() is genes × cells. AnnData wants
# cells × genes, so transpose.
counts = sio.mmread(str(counts_path)).tocsr().T
print(f"Counts: {counts.shape[0]} cells × {counts.shape[1]} genes (sparse, {counts.nnz:,} nnz)")

genes = [line.strip() for line in genes_path.read_text().splitlines() if line.strip()]
barcodes = [line.strip() for line in barcodes_path.read_text().splitlines() if line.strip()]
if len(genes) != counts.shape[1]:
    raise SystemExit(f"genes.tsv ({len(genes)}) doesn't match matrix cols ({counts.shape[1]})")
if len(barcodes) != counts.shape[0]:
    raise SystemExit(f"barcodes.tsv ({len(barcodes)}) doesn't match matrix rows ({counts.shape[0]})")

obs = pd.read_csv(obs_path, index_col=0)
if obs.index.duplicated().any():
    obs = obs[~obs.index.duplicated(keep="first")]
# Align obs to barcode order (R's write.csv preserves row names but
# reordering is cheap insurance).
obs = obs.reindex(barcodes)
print(f"obs: {obs.shape[0]} cells × {obs.shape[1]} columns")
print(f"obs columns: {list(obs.columns)}")

var = pd.DataFrame(index=pd.Index(genes, name="gene_symbol"))
var["gene_symbol"] = var.index

adata = ad.AnnData(X=counts, obs=obs, var=var)
adata.obs_names = barcodes
adata.obs_names_make_unique()
adata.var_names_make_unique()

# Add n_counts for the MaxToki tokenizer.
adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()

print(f"Assembled AnnData: {adata}")
adata.write_h5ad(out_path, compression="gzip")
print(f"Wrote {out_path}")
