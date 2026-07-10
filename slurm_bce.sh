#!/bin/bash
#SBATCH --job-name=peptide-bce
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Plain BCE (CrossEntropyLoss, no class weighting) with ESM2-8M frozen embeddings.
# Baseline to compare against focal loss.

mkdir -p logs

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

echo "===== Plain BCE: sites ====="
python code/esm2_size_experiment.py sites --model-size 8m --loss bce

echo "===== Plain BCE: peptides ====="
python code/esm2_size_experiment.py peptides --model-size 8m --loss bce

echo "All BCE jobs done."
