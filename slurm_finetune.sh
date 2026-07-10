#!/bin/bash
#SBATCH --job-name=peptide-finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

mkdir -p logs checkpoints

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

echo "===== Fine-tune ESM2-8M: sites ====="
python code/finetune_esm2.py sites --epochs 5

echo "===== Fine-tune ESM2-8M: peptides ====="
python code/finetune_esm2.py peptides --epochs 5

echo "===== Post-finetune PCA: sites (fold0, seed0 checkpoint) ====="
CUDA_VISIBLE_DEVICES="" python code/interpret_umap.py \
    --n-samples 2000 \
    --output plots/umap_finetuned \
    --model-path checkpoints/finetune_esm2_8m_sites_fold0_seed0

echo "All fine-tuning jobs done."
