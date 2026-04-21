#!/bin/bash
# Launch the Phase 3b scaled-up senescence KO screen (MAX_CELLS_PER_TYPE=500)
# as a SLURM batch job on Raven. Fully decoupled from any SSH session —
# safe to close the laptop after submission.
#
# Submit from the repo root (so ``slurm-<jobid>.log`` lands there):
#   cd $HOME/maxToki && sbatch run_phase3b.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm-<jobid>.log
#
# Adapt for other phases: change the python script name in the final
# apptainer exec line, and adjust --time if needed.

#SBATCH --job-name=maxtoki-phase3b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.log

set -euo pipefail

module load apptainer

# Ensure output dir exists before the python script tries to write results.
mkdir -p /ptmp/$USER/liver_screen_phase3b

apptainer exec --nv \
    -B $HOME/maxToki:/workspace \
    -B /ptmp/$USER:/ptmp/$USER \
    $HOME/maxToki/maxtoki.sif \
    bash -c "cd /workspace && python phase3_senescence_ko.py"
