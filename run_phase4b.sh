#!/bin/bash
# Launch Phase 4b: multi-anchor ensemble + soft-OE GSN replication.
# Submit from the repo root:
#   cd $HOME/maxToki && sbatch run_phase4b.sh

#SBATCH --job-name=maxtoki-phase4b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.log

set -euo pipefail

module load apptainer

mkdir -p /ptmp/$USER/heart_screen_phase4b

apptainer exec --nv \
    -B $HOME/maxToki:/workspace \
    -B /ptmp/$USER:/ptmp/$USER \
    $HOME/maxToki/maxtoki.sif \
    bash -c "cd /workspace && python phase4b_gsn_heart_ensemble.py"
