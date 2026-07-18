#!/bin/bash
#SBATCH --job-name=peptide-save-model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

mkdir -p logs models

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

echo "===== Saving MLP head weights for app deployment ====="
python code/save_mlp_weights.py

echo "Done. models/sites_head.pt and models/peptides_head.pt saved."
