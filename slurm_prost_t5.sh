#!/bin/bash
#SBATCH --job-name=peptide-prost-t5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

mkdir -p logs

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

# Write large T5 embedding cache to node-local /tmp (scratch quota is full)
export PEPTIDE_CACHE_DIR=/tmp/peptide-cache-$SLURM_JOB_ID
mkdir -p $PEPTIDE_CACHE_DIR

echo "===== prost-t5: sites ====="
python code/esm2_size_experiment.py sites --model-size prost-t5

echo "===== prost-t5: peptides ====="
python code/esm2_size_experiment.py peptides --model-size prost-t5

echo "All prost-t5 jobs done."
