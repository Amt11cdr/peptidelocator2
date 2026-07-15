#!/bin/bash
#SBATCH --job-name=peptide-prot-t5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

mkdir -p logs

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env || true
PYTHON=/home/people/25205761/.conda/envs/peptide-env/bin/python

cd $SLURM_SUBMIT_DIR

# Write large T5 embedding cache to $HOME (scratch quota is full)
export PEPTIDE_CACHE_DIR=$HOME/peptide-cache
mkdir -p $PEPTIDE_CACHE_DIR

echo "===== prot-t5-xl: sites ====="
$PYTHON code/esm2_size_experiment.py sites --model-size prot-t5-xl

echo "===== prot-t5-xl: peptides ====="
$PYTHON code/esm2_size_experiment.py peptides --model-size prot-t5-xl

echo "All prot-t5-xl jobs done."
