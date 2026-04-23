# Convert Ramachandran 2019 tissue.rdata (Seurat object) -> tissue.h5ad.
#
# Run once on the Raven login node:
#   module load R                              # or whatever your R module is
#   cd /ptmp/$USER/liver_fibrosis_phase5
#   Rscript $HOME/maxToki/convert_ramachandran_to_h5ad.R
#
# Requires the Seurat and SeuratDisk packages. Install once if missing:
#   install.packages("Seurat")
#   if (!require("remotes")) install.packages("remotes")
#   remotes::install_github("mojaveazure/seurat-disk")

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
})

in_path  <- "tissue.rdata"
out_h5s  <- "tissue.h5Seurat"
out_h5ad <- "tissue.h5ad"

if (!file.exists(in_path)) {
  stop(sprintf("Cannot find %s in %s. Run the wget step first.", in_path, getwd()))
}

# load() materialises whatever object names the RData saved. We auto-pick
# whichever loaded object is a Seurat object; if multiple, use the first.
loaded <- load(in_path)
cat("Loaded objects:", paste(loaded, collapse = ", "), "\n")

seurat_name <- NULL
for (nm in loaded) {
  obj <- get(nm)
  if (inherits(obj, "Seurat")) { seurat_name <- nm; break }
}
if (is.null(seurat_name)) {
  stop("No Seurat object found in tissue.rdata. Inspect loaded object types manually.")
}
cat(sprintf("Converting Seurat object: %s\n", seurat_name))

# Seurat v5 stores counts in layers; ensure default assay has a countable layer.
obj <- get(seurat_name)
DefaultAssay(obj) <- "RNA"

SaveH5Seurat(obj, filename = out_h5s, overwrite = TRUE)
Convert(out_h5s, dest = "h5ad", overwrite = TRUE)

cat(sprintf("Wrote %s\n", out_h5ad))
