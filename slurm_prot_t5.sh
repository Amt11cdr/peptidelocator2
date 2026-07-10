#!/bin/bash
#SBATCH --job-name=peptide-prot-t5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# ProtT5-XL / ProstT5 are 3B-param models (~12 GB fp32, ~6 GB fp16).
# 48 GB RAM + 32 GB VRAM (V100) should be sufficient.
# Embeddings are computed once and cached to processed-data/*.pqt.

mkdir -p logs

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

echo "===== ProtT5-XL: sites ====="
python code/prot_t5_experiment.py sites --model prot_t5

echo "===== ProtT5-XL: peptides ====="
python code/prot_t5_experiment.py peptides --model prot_t5

echo "===== ProstT5: sites ====="
python code/prot_t5_experiment.py sites --model prost_t5

echo "===== ProstT5: peptides ====="
python code/prot_t5_experiment.py peptides --model prost_t5

echo "All ProtT5 jobs done."
