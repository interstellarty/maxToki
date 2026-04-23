#!/bin/bash
# Launch Phase 5: liver fibrosis KO screen on Ramachandran 2019.
#
# Prerequisites (run once on the login node):
#
#   1. Download the Seurat R object (361 MB):
#        mkdir -p /ptmp/$USER/liver_fibrosis_phase5
#        cd /ptmp/$USER/liver_fibrosis_phase5
#        wget -O tissue.rdata \
#          "https://datashare.ed.ac.uk/bitstream/handle/10283/3433/tissue.rdata?sequence=3&isAllowed=y"
#
#   2. Dump MTX + metadata from the Seurat rdata using R (SeuratObject only
#      — minimal deps, no hdf5r/Shiny cascade). One-time package install:
#        mkdir -p ~/R/library
#        export R_LIBS_USER=~/R/library           # add to ~/.bashrc
#        Rscript -e 'install.packages("SeuratObject", repos="https://cloud.r-project.org")'
#      Then run the dump:
#        module load R
#        cd /ptmp/$USER/liver_fibrosis_phase5
#        Rscript $HOME/maxToki/convert_ramachandran_to_h5ad.R
#      Produces counts.mtx, genes.tsv, barcodes.tsv, obs.csv.
#
#   3. Assemble tissue.h5ad from the MTX files (inside apptainer so scanpy
#      is available):
#        apptainer exec -B $HOME/maxToki:/workspace -B /ptmp/$USER:/ptmp/$USER \
#            $HOME/maxToki/maxtoki.sif \
#            bash -c "cd /ptmp/$USER/liver_fibrosis_phase5 && \
#                     python /workspace/assemble_ramachandran_h5ad.py"
#
#   4. Confirm tissue.h5ad exists:
#        ls -lh /ptmp/$USER/liver_fibrosis_phase5/tissue.h5ad
#
# Then submit:
#        cd $HOME/maxToki && sbatch run_phase5.sh

#SBATCH --job-name=maxtoki-phase5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.log

set -euo pipefail

module load apptainer

mkdir -p /ptmp/$USER/liver_fibrosis_phase5

apptainer exec --nv \
    -B $HOME/maxToki:/workspace \
    -B /ptmp/$USER:/ptmp/$USER \
    $HOME/maxToki/maxtoki.sif \
    bash -c "cd /workspace && python phase5_liver_fibrosis.py"
