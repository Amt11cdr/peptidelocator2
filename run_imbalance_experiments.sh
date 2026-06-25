#!/bin/bash
# Runs all 4 imbalance conditions for both targets (sites and peptides)

cd "$(dirname "$0")"

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
