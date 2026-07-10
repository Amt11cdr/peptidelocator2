#!/bin/bash
#SBATCH --job-name=peptide-prot-t5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

mkdir -p logs

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
source activate peptide-env

cd $SLURM_SUBMIT_DIR

echo "===== prot-t5-xl: sites ====="
python code/esm2_size_experiment.py sites --model-size prot-t5-xl

echo "===== prot-t5-xl: peptides ====="
python code/esm2_size_experiment.py peptides --model-size prot-t5-xl

echo "===== prost-t5: sites ====="
python code/esm2_size_experiment.py sites --model-size prost-t5

echo "===== prost-t5: peptides ====="
python code/esm2_size_experiment.py peptides --model-size prost-t5

echo "All ProtT5 jobs done."
