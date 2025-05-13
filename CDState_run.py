import numpy as np
from pythonScripts import CDState_base as cd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import logging
import copy
import importlib
import scipy
import sklearn, random
import seaborn as sns
import matplotlib.pyplot as plt

path = <path_to_data_directory>

#Load input bulk data
df = pd.read_csv(path + "mixa_bulk_sum.csv", index_col=0,sep=',',header=0)

#Load ground truth source expression
sources = pd.read_csv(path + "mixa_sources_sum.csv", index_col=0,sep=',',header=0)

#Load ground truth source proportions
seta = pd.read_csv(path + "seta_bulk_sum.csv", index_col=0,sep=',',header=0)

sources = sources.loc[df.index,:]
sources = sources.loc[:,["Fibroblasts", "Bcells", "Malignant"]]
seta = seta.loc[:,["Fibroblasts", "Bcells", "Malignant"]]

'''CDState is a non-deterministic method; therefore, we recommend running it multiple times with different initializations.'''

np.random.seed(123)

k=3 #number of states for deconvolution

# create CDState object
cn = cd.CDState(df, num_bases=k, global_round = False, method = "SLSQP")
cn.prepare_data() #filter genes on sex chromosomes and keep only highly variable genes for deconvolution

# cluster input data based on cosine similarity; select one random sample from each cluster as a starting source
cosim = cosine_similarity(cn.data.T)
km = sklearn.cluster.KMeans(n_clusters=k, n_init="auto", init="random").fit(cosim)
km_labels = km.labels_
initial_sources_idx = [random.sample(np.where(km_labels == y)[0].tolist(),1)[0] for y in range(k)]
initial_sources = cn.data[:,initial_sources_idx]
cn.W = copy.copy(initial_sources)

cn.W += 1e-10 #pseudocount to avoid division by 0 in the first iteration
initial_weights = np.random.dirichlet([10]*k, size=cn.data.shape[1])
cn.H = copy.copy(initial_weights.T)
cn.l1 = 1 #first run initial round, where only reconstruction error is optimized
cn.l2 = 0
print("factorizing: alpha=1")
cn.factorize(niter=100000, show_progress = False, err_method="multiobjective")

#use purity information for optimizing global round; l1 and l2 will be optimized automatically
purity = seta.loc[:,'Malignant']
purity.rename(index="purity", inplace=True)
purity.index = df.columns

cnG = cd.CDState(df, purity, global_round=True,  num_bases=k,gene_list = cn.gene_list, method = "SLSQP")
cnG.prepare_data()
cnG.W = copy.copy(cn.W)
cnG.W += 1e-10
cnG.H = copy.copy(cn.H)
cnG.factorize(niter=10000, show_progress = False, err_method="multiobjective")

# compare results with ground tturh:
W_initial = copy.copy(cn.W)
H_initial = copy.copy(cn.H)

W_final = copy.copy(cnG.W)
H_final = copy.copy(cnG.H)


# compare with ground truth sources, initial result:
all_sources = pd.concat([sources.loc[cn.gene_list,:], pd.DataFrame(W_initial, index = cn.gene_list)], axis=1)
all_sources.columns = sources.columns.tolist() + ["S"+str(x) for x in range(1,4)]

corr = all_sources.corr()
cmap = sns.diverging_palette(100, 7, s=75, l=40,
                            n=5, center="light", as_cmap=True)

matrix = np.tril(corr)
sns.heatmap(corr, cmap   = cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, mask=matrix)
plt.show()

#Compare with ground truth weights, initial result:
all_weights = pd.concat([seta, pd.DataFrame(H_initial.T, index = seta.index)], axis=1)
all_weights.columns = sources.columns.tolist() + ["S"+str(x) for x in range(1,4)]

corrW = all_weights.corr()

matrixW = np.tril(corrW)
sns.heatmap(corrW, cmap   = cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, mask=matrixW)
plt.show()

#Compare with ground truth sources, final result:
all_sources = pd.concat([sources.loc[cn.gene_list,:], pd.DataFrame(W_final, index = cn.gene_list)], axis=1)
all_sources.columns = sources.columns.tolist() + ["S"+str(x) for x in range(1,4)]

corr = all_sources.corr()
cmap = sns.diverging_palette(100, 7, s=75, l=40,
                            n=5, center="light", as_cmap=True)

matrix = np.tril(corr)
sns.heatmap(corr, cmap   = cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, mask=matrix)
plt.show()

#Compare with ground truth weights, final result:
all_weights = pd.concat([seta, pd.DataFrame(H_final.T, index = seta.index)], axis=1)
all_weights.columns = sources.columns.tolist() + ["S"+str(x) for x in range(1,4)]

corrW = all_weights.corr()

matrixW = np.tril(corrW)
sns.heatmap(corrW, cmap   = cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, mask=matrixW)
plt.show()
