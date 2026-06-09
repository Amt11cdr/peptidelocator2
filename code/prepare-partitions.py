import os.path as osp
import yaml

import pandas as pd

from hestia.similarity import sequence_similarity_mmseqs
from hestia.partition import ccpart, butina
from sklearn.model_selection import KFold


if __name__ == "__main__":
    df = pd.read_csv("processed-data/peptide.tsv", sep='\t')
    df['original-prot'] = df.index.to_numpy()
    df2 = pd.read_parquet("processed-data/peptide.pqt")

    sim_df = sequence_similarity_mmseqs(
        df,
        field_name='Sequence',
        denominator='n_aligned',
        prefilter=False,
        verbose=3,
        threshold=0.1
    )
    for th_out in range(10, 100, 10):
        print("CCPart: ", th_out)
        train_idx, test_idx, _ = ccpart(
            df=df,
            sim_df=sim_df,
            threshold=th_out/100
        )
        if len(test_idx) >= len(df) * 0.185:
            break

    train_df = df.iloc[train_idx, :].reset_index(drop=True)
    test_df = df.iloc[test_idx, :].reset_index(drop=True)
    df2.loc[df2['original-prot'].isin(test_df['original-prot']), 'split'] = 'test'
    df2.loc[df2['original-prot'].isin(train_df['original-prot']), 'split'] = 'train'

    all_metadata = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for idx, (train_idx, test_idx) in enumerate(kfold.split(train_df)):
        # train_df = df2[df2['original-prot'].isin(train_idx)]
        train_t_df = train_df.iloc[train_idx, :].reset_index(drop=True)
        sim_df = sequence_similarity_mmseqs(
            train_t_df,
            field_name='Sequence',
            denominator='n_aligned',
            prefilter=False,
            verbose=3,
            threshold=0.1
        )
        for th in range(20, 100, 10):
            train, test, _ = butina(
                df=train_t_df,
                sim_df=sim_df,
                threshold=th/100,
            )
            if len(test) >= len(train_t_df) * 0.185:
                break

        train_train = train_t_df.iloc[train, :]
        test_test = train_t_df.iloc[test, :]

        df2.loc[df2['original-prot'].isin(train_train['original-prot']), f'fold-{idx}'] = 'train'
        df2.loc[df2['original-prot'].isin(test_test['original-prot']), f'fold-{idx}'] = 'test'
        df2.loc[df2['original-prot'].isin(test_idx), f'fold-{idx}'] = 'valid'
        metadata = {"cv-partitioning-algorithm": "butina", "cv-fold": idx, "cv-min-th": th}
        all_metadata.append(metadata)
    all_metadata.append({"test-partitioning-algorithm": "ccpart", "min-th": th_out})
    df2.to_parquet("processed-data/peptide-partitions.pqt")
    yaml.safe_dump(all_metadata, open("results/partitioning-metadata.yml", "w"))
