#!/bin/bash
#SBATCH --job-name=peptide-interpret
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8

mkdir -p logs

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

echo "===== UMAP Analysis ====="
CUDA_VISIBLE_DEVICES="" python code/interpret_umap.py --n-samples 2000 --output plots/umap

echo "===== Attention Analysis ====="
CUDA_VISIBLE_DEVICES="" python code/interpret_attention.py --window 15 --output plots/attention

echo "All interpretability analyses done."
