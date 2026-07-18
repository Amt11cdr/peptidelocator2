# PeptideLocator2

Predicts cleavage sites and peptide regions in protein sequences using ESM2 protein language model embeddings.

## Run the app

```bash
docker run -p 7860:7860 amt11cdr/peptidelocator2:v2
```

Open `http://localhost:7860` in your browser.

## Results

| Model | Sites MCC | Peptides MCC |
|---|---|---|
| ESM2-8M (frozen) | 0.551 ± 0.033 | 0.739 ± 0.016 |
| ESM2-150M (frozen) | 0.624 ± 0.041 | 0.789 ± 0.017 |
| ESM2-650M (frozen) | 0.575 ± 0.235 | 0.759 ± 0.077 |

Evaluated with 5-fold × 5-seed cross-validation using similarity-aware splits.

## UCD Shields Lab
