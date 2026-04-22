#!/bin/bash
# Launch Phase 4c: GSN KO on MaxToki-1B (capacity check).
#
# Before first submission, download the 1B HF checkpoint on the login node:
#   module load apptainer
#   apptainer exec -B $HOME/maxToki:/workspace -B /ptmp/$USER:/ptmp/$USER \
#       $HOME/maxToki/maxtoki.sif \
#       huggingface-cli download theodoris-lab/MaxToki \
#         --include "MaxToki-1B-HF/*" \
#         --local-dir /ptmp/$USER/models/maxtoki-hf
#
# Then submit:
#   cd $HOME/maxToki && sbatch run_phase4c.sh

#SBATCH --job-name=maxtoki-phase4c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.log

set -euo pipefail

module load apptainer

mkdir -p /ptmp/$USER/heart_screen_phase4c

apptainer exec --nv \
    -B $HOME/maxToki:/workspace \
    -B /ptmp/$USER:/ptmp/$USER \
    $HOME/maxToki/maxtoki.sif \
    bash -c "cd /workspace && python phase4c_gsn_heart_1b.py"
