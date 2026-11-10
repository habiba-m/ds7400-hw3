#!/bin/bash
# HW3: one experiment = train the autoencoder with a given supernode_alpha, then score it.
#   sbatch -J hw3_a0.0 slurm/hw3.slurm 0.0
#SBATCH -A sds_baek_energetic
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load miniforge
A=$1
OUT=/scratch/crv4jj/parc3d_v2/runs/hw3/alpha_$A
python -u train_ae.py --config configs/hw3.json --set out=$OUT model.supernode_alpha=$A
# metrics on every cell of the 9 test runs, every 10th snapshot
python -u evaluate.py --config configs/hw3.json --set out=$OUT model.supernode_alpha=$A \
    --modes recon --no-gif --frame-stride 10
# one GIF on the never-seen mesh for the video (written to a separate folder so metrics.csv is kept)
python -u evaluate.py --config configs/hw3.json --set out=$OUT/gif model.supernode_alpha=$A \
    --ae $OUT/ae/best.pt --modes recon --runs B_m02_s11_yawstep --frame-stride 3
