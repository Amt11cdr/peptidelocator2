import os
import yaml

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import typer

from evaluate import evaluate_model


def run_experiment(target: str):
    df = pd.read_parquet("processed-data/peptide-partitions.pqt")
    df['y'] = df[f'y-{target}']
    os.makedirs('results', exist_ok=True)
    metadata = {
        "model": "rf",
        "hpo": "no",
        "downsampling": "no",
    }
    all_results = []
    for seed in range(5):
        for fold in range(5):
            print(f"Running Fold {fold} with seed {seed}")
            rf = RandomForestClassifier(
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1
            )
            train_df = df[df[f'fold-{fold}'] == 'train']
            train_df = pd.concat([
                train_df[train_df['y'] == 0].sample(
                    frac=1.0
                    # int(train_df['y'].sum() * 1.5), random_state=seed
                ),
                train_df[train_df['y'] == 1]
            ])
            test_df = df[df[f'fold-{fold}'] == 'test']

            rf.fit(np.stack(train_df['x']), train_df['y'])
            preds = rf.predict(np.stack(test_df['x']))
            preds_proba = rf.predict_proba(np.stack(test_df['x']))
            result = evaluate_model(
                y_true=test_df['y'],
                y_pred=preds,
                y_pred_proba=preds_proba
            )
            result['fold'] = fold
            result['seed'] = seed
            print(f"MCC: {result['mcc'].item():.2f}")
            all_results.append(result)
    result_df = pd.concat(all_results)
    result_df.to_csv(f"results/results-{target}-predictor-1.csv", index=False)
    yaml.safe_dump(metadata, open(f"results/{target}-1.yml", 'w'))
    print(f"MCC: {result_df['mcc'].mean():.2f}±{result_df['mcc'].std():.2f}")
    print(f"Weighted precision: {result_df['precision_weighted'].mean():.2f}±{result_df['precision_weighted'].std():.2f}")
    print(f"Recall: {result_df['recall_macro'].mean():.2f}±{result_df['recall_macro'].std():.2f}")


if __name__ == "__main__":
    typer.run(run_experiment)