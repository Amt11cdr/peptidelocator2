#!/bin/bash
#SBATCH --job-name=peptide-imbalance
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Load anaconda and activate environment
module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
conda activate peptide-env

# Move to repo directory (replace with your actual path on Sonic)
cd $SLURM_SUBMIT_DIR

for target in sites peptides; do
    echo "===== Target: $target ====="

    echo "-- Focal, no downsample --"
    python code/mlp_experiment.py $target --loss-type focal --no-downsample

    echo "-- Focal + downsample --"
    python code/mlp_experiment.py $target --loss-type focal --downsample

    echo "-- BCE, no downsample --"
    python code/mlp_experiment.py $target --loss-type bce --no-downsample

    echo "-- BCE + downsample --"
    python code/mlp_experiment.py $target --loss-type bce --downsample
done

echo "All experiments done."
