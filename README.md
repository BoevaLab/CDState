# CDState
### An unsupervised approach to predict malignant cell heterogeneity in tumor bulk RNA-sequencing data

[![Preprint](https://img.shields.io/badge/preprint-available-green)](https://doi.org/10.1101/2025.03.01.641017) &nbsp;


CDState is an unsupervised deconvolution method for tumor bulk RNA-sequencing data, aimed at identifying malignant cell states and their proportions.

## Usage

### Installation with Conda

1. Clone the repository:
   ```bash
   git clone https://github.com/BoevaLab/CDState.git
   cd CDState
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
