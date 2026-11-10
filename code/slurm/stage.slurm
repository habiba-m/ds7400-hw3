#!/bin/bash
# One stage on one A100.   sbatch slurm/stage.slurm sanity.py configs/pilot.json
#SBATCH -A sds_baek_energetic
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load miniforge
python -u "$1" --config "$2" "${@:3}"
