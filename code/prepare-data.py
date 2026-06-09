import os.path as osp

from typing import List

import numpy as np
import pandas as pd
import typer

from autopeptideml.reps.lms import RepEngineLM


def process_peptide(sentence: str) -> str:
    """
    Process Peptide string and output start and end points
    for a given peptide
    """
    frags = sentence.split(";")
    frags = [f.replace("PEPTIDE ", "").replace("..", "-")
             for f in frags if "PEPTIDE" in f]
    return ";".join(frags)


def cleavage_mask(sites: List[int], sequence: str) -> List[int]:
    """"
    Guided by the indices in the list of cleavage points create a mask
    where there is a "1" for residues with a cleavage point and a 0
    for all the rest
    """
    mask = np.zeros(len(sequence))
    for s in sites:
        mask[s-1] = 1
    return mask


def peptide_mask(sites: List[int], sequence: str) -> List[int]:
    """"
    Guided by the indices in the list of peptides create a mask
    where there is a "1" for residues with a cleavage point and a 0
    for all the rest
    """
    mask = np.zeros(len(sequence))
    for s in sites:
        mask[s[0]:s[1]] = 1
    return mask


SEQUENCE_ORDER = 0


def origin_mask(sequence: str) -> List[int]:
    global SEQUENCE_ORDER
    mask = np.ones(len(sequence))
    mask *= SEQUENCE_ORDER
    SEQUENCE_ORDER += 1
    return mask


def unravel(array: List[np.ndarray]) -> np.ndarray:
    new_array = []
    for array in array:
        for row in array:
            new_array.append(row)
    new_array = np.stack(new_array)
    return new_array


def run():
    df = pd.read_csv("data/swissprot-peptide.tsv", sep='\t')
    df = df[['Entry', "Organism", "Length", "Peptide", "Sequence"]]
    df["Peptide"] = df["Peptide"].map(process_peptide)
    df.to_csv("processed-data/peptide.tsv", sep='\t', index=False)
    df['Sites'] = df['Peptide'].map(lambda x: x.split(";"))
    sites, peptides = [], []
    for peptide, length in zip(df['Sites'], df['Length']):
        sites.append([])
        peptides.append([])
        for site in peptide:
            if '>' in site or '<' in site or "?" in site:
                continue
            else:
                start = int(site.split('-')[0])
                end = int(site.split('-')[1])
                if start != 1:
                    sites[-1].append(start - 1)
                if end != length:
                    sites[-1].append(end + 1)
                peptides[-1].append((start, end))

    df['Peptides'], df['Sites'] = peptides, sites
    print(f"Num of proteins without sites: {len(df[df['Sites'].map(len) < 1]):,}")
    df = df[df['Sites'].map(len) > 0]
    long_seqs = len(df[df['Length'] > 1022])
    print(f"There are {long_seqs:,} sequences that are too long")
    df = df[df['Sequence'].map(len) <= 1022]
    df.to_parquet("processed-data/peptide.pqt", index=False)

    site_masks = df.apply(
        lambda x:
        cleavage_mask(x['Sites'], x['Sequence']),
        axis=1
    )
    peptide_masks = df.apply(
        lambda x:
        peptide_mask(x['Peptides'], x['Sequence']),
        axis=1
    )
    re = RepEngineLM('esm2-8m', average_pooling=False)
    re.move_to_device("mps")
    reps = re.compute_reps(
        df['Sequence'], batch_size=64
    )
    new_origins = unravel(df['Sequence'].map(origin_mask))
    new_site_masks = unravel(site_masks)
    new_peptide_masks = unravel(peptide_masks)
    new_reps = unravel(reps)
    df = pd.DataFrame()
    df['x'] = [r.tolist() for r in new_reps]
    df['original-prot'] = [o for o in new_origins]
    df['y-sites'] = [m for m in new_site_masks]
    df['y-peptides'] = [m for m in new_peptide_masks]
    df.to_parquet("processed-data/peptide.pqt")


if __name__ == "__main__":
    run()
