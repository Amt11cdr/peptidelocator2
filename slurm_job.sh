#!/bin/bash
#SBATCH --job-name=peptide-imbalance
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

python code/mlp_experiment.py peptides --loss-type focal --no-downsample
python code/mlp_experiment.py peptides --loss-type focal --downsample
python code/mlp_experiment.py peptides --loss-type bce --no-downsample
python code/mlp_experiment.py peptides --loss-type bce --downsample
python code/mlp_experiment.py sites --loss-type bce --no-downsample
python code/mlp_experiment.py sites --loss-type bce --downsample

echo "Done."
