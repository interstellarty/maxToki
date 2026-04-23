# Convert Ramachandran 2019 tissue.rdata -> MTX + metadata files that Python
# can assemble into an h5ad without hdf5r or SeuratDisk.
#
# This path uses ONLY SeuratObject (minimal deps, ~2 min install) plus base-R
# Matrix. No Shiny/rmarkdown/bslib/hdf5r cascade.
#
# One-time install (Raven user library):
#   mkdir -p ~/R/library
#   export R_LIBS_USER=~/R/library      # add to ~/.bashrc for persistence
#   Rscript -e 'install.packages("SeuratObject", repos="https://cloud.r-project.org")'
#
# Run (login node, after downloading tissue.rdata):
#   module load R
#   cd /ptmp/$USER/liver_fibrosis_phase5
#   Rscript $HOME/maxToki/convert_ramachandran_to_h5ad.R
#
# Outputs (in the current working directory):
#   counts.mtx           sparse counts matrix (genes × cells)
#   genes.tsv            one gene symbol per line (var_names)
#   barcodes.tsv         one cell id per line (obs_names)
#   obs.csv              cell metadata (condition, donor, annotation, ...)
#
# Then run the companion Python script to assemble tissue.h5ad:
#   python $HOME/maxToki/assemble_ramachandran_h5ad.py

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
})

in_path <- "tissue.rdata"
if (!file.exists(in_path)) {
  stop(sprintf("Cannot find %s in %s. Run the wget step first.", in_path, getwd()))
}

loaded <- load(in_path)
cat("Loaded objects:", paste(loaded, collapse = ", "), "\n")

# Pick whichever loaded object inherits from Seurat.
seurat_name <- NULL
for (nm in loaded) {
  obj <- get(nm)
  if (inherits(obj, "Seurat")) { seurat_name <- nm; break }
}
if (is.null(seurat_name)) {
  stop("No Seurat object found in tissue.rdata. Inspect loaded object types manually.")
}
cat(sprintf("Exporting Seurat object: %s\n", seurat_name))

obj <- get(seurat_name)
DefaultAssay(obj) <- "RNA"

# Extract counts. `GetAssayData` works across Seurat v3/v4/v5.
counts <- GetAssayData(obj, assay = "RNA", layer = "counts")
if (is.null(counts) || length(counts) == 0) {
  # Older Seurat API uses `slot` instead of `layer`.
  counts <- GetAssayData(obj, assay = "RNA", slot = "counts")
}
cat(sprintf("Counts matrix: %d genes x %d cells (sparse = %s)\n",
            nrow(counts), ncol(counts), inherits(counts, "dgCMatrix")))

writeMM(as(counts, "CsparseMatrix"), "counts.mtx")
writeLines(rownames(counts), "genes.tsv")
writeLines(colnames(counts), "barcodes.tsv")
write.csv(obj@meta.data, "obs.csv", row.names = TRUE)

cat("Wrote: counts.mtx, genes.tsv, barcodes.tsv, obs.csv\n")
