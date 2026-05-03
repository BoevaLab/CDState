To run the tutorial you need to install Jupyter Notebook in cdstate conda environment. Run:

`conda install -c conda-forge notebook`

Then run:
`jupyter notebook` and select `CDState_*.ipynb` which you want to use.

You will also need matplotlib:
`python -m pip install --upgrade pip`

`python -m pip install matplotlib`

To run CDState on your samples, you need purity values. If you do not have them already, you can get them using RNA-seq-based methods, e.g., ESTIMATE []() or PUREE [](). The tutorial contains code snippets to do this using ESTIMATE, but it requires R to be installed in the enrivonment. If you do not have R, run:

`conda activate cdstate`

`conda install -c conda-forge r-base rpy2`

