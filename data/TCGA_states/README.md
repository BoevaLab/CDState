# Malignant cell states identified by CDState in TCGA data

CState was applied to 33 cancer types from The Cancer Genome Atlas (TCGA) to characterize malignant cell states using bulk RNA-seq data. Each state has been characterized using enriched hallmark gene sets from MSigDB. These annotations can be found in `States_enriched_programs.csv`.

For each cancer type, there are two files in the directory:
- `*_states.csv` : with expression of identified malignant cell states shared across samples from the particular cancer type [genes x states]
` '*proportions.csv` : with relative proportions of these states, and the calculated ITTH index for each input sample [samples x (states and ITTH)]

For details about the states see [Kraft et al.](https://www.biorxiv.org/content/10.1101/2025.03.01.641017v3).
