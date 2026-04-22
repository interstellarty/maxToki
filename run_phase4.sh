#!/bin/bash
# Launch the Phase 4 GSN-KO cardiac-fibroblast replication screen as a SLURM
# batch job on Raven. Safe to close the laptop after submission.
#
# Submit from the repo root (so ``slurm-<jobid>.log`` lands there):
#   cd $HOME/maxToki && sbatch run_phase4.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm-<jobid>.log

#SBATCH --job-name=maxtoki-phase4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.log

set -euo pipefail

module load apptainer

mkdir -p /ptmp/$USER/heart_screen_phase4

apptainer exec --nv \
    -B $HOME/maxToki:/workspace \
    -B /ptmp/$USER:/ptmp/$USER \
    $HOME/maxToki/maxtoki.sif \
    bash -c "cd /workspace && python phase4_gsn_heart.py"
