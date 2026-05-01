# CDState
### An unsupervised approach to predict malignant cell heterogeneity in tumor bulk RNA-sequencing data

[![Preprint](https://img.shields.io/badge/preprint-available-green)](https://doi.org/10.1101/2025.03.01.641017) &nbsp;


CDState is an unsupervised deconvolution method for tumor bulk RNA-sequencing data, aimed at identifying malignant cell states and their proportions.

## Usage

### Installation with Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-user>/cdstate.git
   cd cdstate
   ```

2. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate cdstate
   ```

4. Verify Python version:
   ```bash
   python --version
   ```

Expected version:
```bash
Python 3.10.x
```

## Updating the environment

If dependencies change, run:
```bash
conda env update -f environment.yml --prune
```

## Basic usage

Input bulk data should have genes in rows, samples in columns.

Input purity should have a column 'purity', with samples in rows.


```python
import CDState_base as cd
import pandas as pd
import copy
import numpy as np

data = pd.read_csv("data/bulkified_mixes/mixa_bulk_sum.csv", index_col=0,sep=',',header=0)
proportions = pd.read_csv("data/bulkified_mixes/seta_bulk_sum.csv", index_col=0,sep=',',header=0)

purity = proportions.loc[:,'Malignant']
purity.rename(index="purity", inplace=True)
purity.index = data.columns
```
Create CDState object:
```python
k = 3 # number of sources
cn = cd.CDState(data, num_bases=k, global_round = False)
```
Prepare data - filter out genes from sex chromosomes and keep only highly variable genes for deconvolution:
```python
cn.prepare_data() 
```
Initialize sources as random k samples after gene filtering:
```python
n_cols = cn.data.shape[1]
cols = np.random.choice(n_cols, size=k, replace=False)
initial_sources = cn.data[:, cols]
cn.W = copy.copy(initial_sources)
cn.W += 1e-10 # add pseudocount to avoid division by 0
```

Run Step 1:
```python
cn.factorize()
```

Run Step 2:
```python
cnG = cd.CDState(data, purity, num_bases=k, global_round = True, gene_list = cn.gene_list)
cnG.H = copy.copy(cn.H) # start from proportions found in Step 1
cnG.W = copy.copy(cn.W) # start from sources found in Step 1
cnG.prepare_data()
cnG.factorize()
```


## Citing CDState
If you use CDState in your work, you can cite it using
```BibTex
@article {Kraft2025.03.01.641017,
	author = {Kraft, Agnieszka and Yates, Josephine and Barkmann, Florian and Boeva, Valentina},
	title = {CDState: an unsupervised approach to predict malignant cell heterogeneity in tumor bulk RNA-sequencing data},
	elocation-id = {2025.03.01.641017},
	year = {2025},
	doi = {10.1101/2025.03.01.641017},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/04/09/2025.03.01.641017},
	eprint = {https://www.biorxiv.org/content/early/2025/04/09/2025.03.01.641017.full.pdf},
	journal = {bioRxiv}
}
```

## License
CDState is licensed under the MIT License. See the `LICENSE` file for details.
