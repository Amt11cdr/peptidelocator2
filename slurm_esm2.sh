#!/bin/bash
#SBATCH --job-name=peptide-esm2-sizes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

mkdir -p logs

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

# ESM2-8M (embeddings already computed, fastest)
echo "===== ESM2-8M ====="
for target in sites peptides; do
    echo "-- $target --"
    python code/esm2_size_experiment.py $target --model-size 8m
done

# ESM2-150M (recomputes embeddings on first run, caches for second)
echo "===== ESM2-150M ====="
for target in sites peptides; do
    echo "-- $target --"
    python code/esm2_size_experiment.py $target --model-size 150m
done

# ESM2-650M (largest model, slowest embedding computation)
echo "===== ESM2-650M ====="
for target in sites peptides; do
    echo "-- $target --"
    python code/esm2_size_experiment.py $target --model-size 650m
done

echo "All ESM2 size experiments done."
