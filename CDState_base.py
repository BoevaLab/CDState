import numpy as np
import pandas as pd
import logging
import logging.config
import scipy.sparse
import scipy.optimize, scipy.stats 
from cvxopt import solvers, matrix
import time
import copy
import math
import sklearn.metrics
from numpy.linalg import norm
from scipy.optimize import nnls
import itertools
from itertools import chain, combinations
import jax
import jax.numpy as jnp
from jax import grad
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.utils import resample


_EPS = 0.0001 #like in python decomposition.NMF


class CDState():
  
	_EPS = _EPS

	def __init__(self, data, purity=None, global_round = False, num_bases=4, method = None, l1=1, l2=0, lr = 0, threshold_low = 0.3, threshold_high = 0.99, gene_list=None, fixed=None, **kwargs):

		def setup_logging():
				self._logger = logging.getLogger("cdstate")
				if not self._logger.handlers:
					console_handler = logging.StreamHandler()
					console_handler.setLevel(logging.DEBUG)
					log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
					console_handler.setFormatter(log_format)
					self._logger.addHandler(console_handler)

		setup_logging()

		# set variables
		self.samples = data.columns
		self.gene_list = gene_list
		self.threshold_low = threshold_low
		self.threshold_high = threshold_high
		self.raw_data = data.copy() #store raw df
		self._num_bases = num_bases
		self.method = method
		self.l1 = l1
		self.l2 = l2
	   
		self.purity = purity
		self.keep = False 
		self.global_round = global_round #by default assume it is initial round, l2 = 0
		if (self.global_round == True and self.purity is None):
				self._logger.error("You must provide purity for global round!")
		
		self.cons = {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}
		self.bounds = [[0., 1.]]*self._num_bases # or None instead of 1

		
	def prepare_data(self, path_genes="gene_order.csv"): #provide a path to the file with gender-related genes
		if (self.gene_list != None):
			self.data = self.raw_data.loc[self.gene_list,:].to_numpy()
			
		else:
			genes = pd.read_csv(path_genes, sep=',', header=0)
			sex_g = genes[genes['chromosome'].isin(["chrX", "chrY"])]['gene_name'].tolist()
			data_autosomes = self.raw_data.copy() #pandas df
			#filter out genes on sex chromosomes
			data_autosomes = data_autosomes[~data_autosomes.index.isin(sex_g)]
			#fitler out not expressed genes
			data_autosomes = data_autosomes.loc[data_autosomes.sum(axis=1) > 0]
			tmp_data = data_autosomes.T
			#calculate Frobenius norm and standard deviation for each gene
			sigNorm = np.linalg.norm(tmp_data, axis=0) #calculates the magnitude of given gene across all samples
			sds = tmp_data.std(axis=0) #caluclates SD of each gene
			#create DataFrame with log-transformed norms and SDs
			df_tmp = pd.DataFrame({'L2': np.log10(sigNorm),'SD': np.log10(sds)}, index=tmp_data.columns)
			#apply filtering based on L2 norm thresholds
			df_tmp["group"] = None
			L2_lower = np.log10(np.quantile(sigNorm, self.threshold_low)) #calculate thresholds for quantiles, use as parameter
			L2_upper = np.log10(np.quantile(sigNorm, self.threshold_high))
			df_tmp = df_tmp[(df_tmp['L2'] > L2_lower) & (df_tmp['L2'] <= L2_upper)] #already filtered given thresholds
			#split genes into bins based on L2 values
			bin_seq = np.arange(df_tmp['L2'].min(), df_tmp['L2'].max(), 0.1)
			bin_seq[0] = df_tmp['L2'].min() - 1
			bin_seq[-1] = df_tmp['L2'].max()
			df_tmp['new_bin'] = pd.cut(df_tmp['L2'], bins=bin_seq) #split genes into bins
			#filter top half of genes by SD within each bin
			filtered_genes = []
			for b in df_tmp['new_bin'].unique():
				tmp = df_tmp[df_tmp['new_bin'] == b].sort_values('SD', ascending=False)
				filtered_genes.extend(tmp.index[:round(len(tmp) / 2)])
			self.gene_list = filtered_genes
			filtered_data = data_autosomes.loc[filtered_genes]
			self.data = filtered_data.to_numpy()
		self._data_dimension, self._num_samples = self.data.shape
		return

	def residual(self):
		res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
		total = 100.0*res/np.sum(np.abs(self.data))
		return total

	def frobenius_norm(self):
		# check if W and H exist
		if hasattr(self,'H') and hasattr(self,'W'):
				reconstructed = np.dot(self.W, self.H)
				tmp_err = (self.data[:,:] - reconstructed)**2
				err = (np.sum(tmp_err))
		else:
				err = None

		return err
		
	def jaxcosim(self, i1, i2):
		x1 = jnp.array(self.W[:,i1])
		x2 = jnp.array(self.W[:,i2])
		return jnp.dot(x1, x2)/(jnp.linalg.norm(x1)*jnp.linalg.norm(x2))

	def calculate_cosine(self):
		cosSim = 0
		for pair in list(itertools.combinations(range(self._num_bases), 2)):
			cosSim += self.jaxcosim(pair[0], pair[1])
		return cosSim
		
	def calculate_mmse(self):
		if len(self.mal)==1:
			predicted = self.H.T[:,self.mal]
		else:
			predicted = np.sum(self.H.T[:,self.mal], axis=1)
		mse = sklearn.metrics.mean_squared_error(predicted, self.purity.to_numpy())
		return mse

	def find_malignant(self):
		H = pd.DataFrame(self.H.T, index = self.samples)
		tmp = pd.concat([self.purity, H], axis=1)
		# Calculate Spearman's correlations between purity and each column of H
		correlations = {}
		p_values = {}
		for column in H.columns:
			r, p = spearmanr(tmp['purity'], H[column])
			correlations[column] = r
			p_values[column] = p
		# Sort by correlation in decreasing order
		sorted_columns = sorted(correlations, key=correlations.get, reverse=True)
		sorted_r = [correlations[col] for col in sorted_columns]
		sorted_p = [p_values[col] for col in sorted_columns]
		# Initialize lists for aggregated correlations and p-values
		cors = []  # Stores correlation with aggregated values
		ps = []	# Respective p-values for above coefficients
		# Greedy search
		cors.append(sorted_r[0])
		ps.append(sorted_p[0])
		for i in range(1, len(sorted_columns)-1):
			sel = sorted_columns[:i + 1]
			aggregated = H[sel].sum(axis=1)
			r, p = spearmanr(aggregated, tmp['purity'])
			cors.append(r)
			ps.append(p)
		# Exclude non-significant results
		cors = [r if p / 2 <= 0.05 else -np.inf for r, p in zip(cors, ps)]
		# Find the best subset
		if (max(cors) == -np.inf):
			mal = []
		else:
			best = np.argmax(cors)
			mal = sorted_columns[:best + 1]
		self.mal = mal
		return


	def find_malignant_bootstrap(self):
		'''Run if less than 20 samples are used for deconvolution'''
		mal = []  # Initialize an empty list to store indices
		H = pd.DataFrame(self.H.T, index = self.samples)
		tmp = pd.concat([self.purity, H], axis=1)
		tmp.columns = ['purity'] + list(H.columns)
		# Calculate Spearman's correlations and p-values between purity and each column in H
		correlations = {}
		p_values = {}
		for col in H.columns:
			corr, pval = spearmanr(tmp['purity'], tmp[col])
			correlations[col] = corr
			p_values[col] = pval
		# Sort correlations in decreasing order
		sorted_columns = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)
		sorted_correlations = [correlations[col] for col in sorted_columns]
		sorted_p_values = [p_values[col] for col in sorted_columns]
		cors = []  # Store correlation with aggregated values
		ps = []	# Respective p-values for the above coefficients
		# Greedy search
		for i in range(len(sorted_columns)-1):
			bootstrap_correlations = []  # Initialize bootstrap correlations
			y = tmp['purity'].to_numpy()
			if i == 0:
				sel = [sorted_columns[i]]
				x = H[sel[0]].to_numpy()
			else:
				sel = sorted_columns[:i+1]
				x = H[sel].sum(axis=1).to_numpy()
			for _ in range(1000):
				resample_indices = np.random.choice(len(x), size=len(x), replace=True)
				x_resampled = x[resample_indices]
				y_resampled = y[resample_indices]
				corr, _ = spearmanr(x_resampled, y_resampled)
				bootstrap_correlations.append(corr)
			bootstrap_correlations = [x for x in bootstrap_correlations if not math.isnan(x)]
			lower_bound = np.percentile(np.array(bootstrap_correlations), 2.5)
			ps.append(lower_bound)
			cors.append(np.mean(bootstrap_correlations))
		# Exclude non-significant values
		cors = [c if p > 0 else -np.inf for c, p in zip(cors, ps)]
		best = np.argmax(cors)  # Find index of the maximum correlation
		mal = sorted_columns[:best+1]
		self.mal = mal
		return
			
	def infer_full(self):
		'''Function to infer expression of the full input gene list'''
		if hasattr(self,'H') and hasattr(self,'W'):
			#use estimated matrix H to infer gene expression of the filtered out genes using nnls
			filtered = self.raw_data[~self.raw_data.index.isin(self.gene_list)].copy()
			W_filtered = np.zeros((filtered.shape[0], self._num_bases))
			#iterate for each gene:
			for i in range(filtered.shape[0]):
				W_filtered[i, :], _ = nnls(self.H.T, filtered.iloc[i, :].to_numpy())
			W_filtered = pd.DataFrame(W_filtered, index = filtered.index)
			W = pd.DataFrame(self.W, index = self.gene_list)
			W_filtered.columns = W.columns
			self.full_W = pd.concat([W, W_filtered], axis=0)
			return
				
	def multi_objective(self):
		# check if W and H exist
		if hasattr(self,'H') and hasattr(self,'W'):
				reconstructed = np.dot(self.W, self.H)
				tmp_err = (self.data[:,:] - reconstructed)**2
				err1 = (np.sum(tmp_err)) #frobenius
				err2 = self.calculate_cosine()
				#print("errors below:") #if false then still ok, they are almost the same, numerical stability
				#print(err1, err2, self.l1*err1+self.l2*self.scaler*err2)
				err = self.l1*err1 + self.l2*self.scaler*err2

		else:
				err = None
				err1 = None
				err2 = None

		return err, err1, err2

	def _init_w(self):
		self.W = np.random.random((self._data_dimension, self._num_bases)) + 10**-10

	def _init_h(self):
		self.H = np.full((self._num_bases, self._num_samples), 1./self._num_bases)

	def _update_h(self):
		pass

	def _update_w(self):
		pass

	def _converged(self, i):
		derr = np.abs(self.ferr[i-1] - self.ferr[i]) / np.abs(self.ferr[i-1])
		if self.global_round == True:
			if self.keep == False:
				if (self.mmse[i]< self.mmse[i-1] and derr < 10*self._EPS):
					if round(self.l2,2) < 0.9:
						self.l2 = round(self.l2 + 0.1,2)
					else:
						self.keep = True
						#print("Max beta")
					self.l1 = round(1 - self.l2,2)
				elif (self.mmse[i] > self.mmse[i-1]):
					self.keep = True
					#print("self.keep == True")
					if round(self.l2,2) >= 0.1:
						self.l2 = round(self.l2 - 0.1,2)
					self.l1 = round(1 - self.l2,2)
			else:
				if round(self.l2,2) >= 0.1:
					if derr < 10*self._EPS:
						self.l2 = round(self.l2 - 0.1,2)
						self.l1 = round(1 - self.l2, 2)
				else:
					if derr < self._EPS:
						return True
			return False
		else:
			if derr < self._EPS:
				return True
			else:
				return False



	def factorize(self, niter=100, show_progress=True,
						compute_w=True, compute_h=True, compute_err=True, err_method = "multiobjective"):
		if show_progress:
				self._logger.setLevel(logging.INFO)
		else:
				self._logger.setLevel(logging.ERROR)

		if not hasattr(self,'W'):# and compute_w:
				print("Initializing W")
				self._init_w()

		if not hasattr(self,'H'):# and compute_h:
				print("Initializing H")
				self._init_h()

		if compute_err:
			self.ferr = np.zeros(niter+1)
			self.err1 = np.zeros(niter+1)
			self.err2 = np.zeros(niter+1)
			
		starting_error1 = self.frobenius_norm()
		starting_error2 = self.calculate_cosine()
		self.scaler = starting_error1 / starting_error2
		self.ferr[0] = self.l1*starting_error1 + self.l2*self.scaler*starting_error2
		print("Starting multiobjective error: ", self.ferr[0])
		self.err1[0] = starting_error1
		self.err2[0] = starting_error2

		if self.global_round == True:
				if self._num_samples >=20:
						self.find_malignant()
				else:
						self.find_malignant_bootstrap()
				self.l2 = 0.1
				self.l1 = round(1 - self.l2,2)
				self.mmse = np.full(niter+1, np.inf)
				if self.mal != []:
						self.mmse[0] = self.calculate_mmse() 
				self.betas = np.full(niter+1, 0)
				self.betas[0] = self.l2
				
		start_time = time.perf_counter()
		for i in range(1,niter+1):
				if compute_h:
					print("updating H first")
					self._update_h()

				if compute_w: 
					self._update_w()
				
				if self.global_round == True:

					if self._num_samples >=20:
						self.find_malignant()
					else:
						self.find_malignant_bootstrap()
					#check if malignant signal is found:
					if self.mal == []:
						self.ferr = self.ferr[:i]
						self.err1 = self.err1[:i]
						self.err2 = self.err2[:i]
						self.mmse = self.mmse[:i]
						self.betas = self.betas[:i]
						print("Cannot identify malignant source for global optimization, terminating")
						break
					self.mmse[i] = self.calculate_mmse()
					self.betas[i] = self.l2
					self.find_malignant()
					print("Self.mal:")
					print(self.mal)

				if compute_err:
					multi_error = self.multi_objective()
					self.ferr[i] = multi_error[0]
					self.err1[i] = multi_error[1]
					self.err2[i] = multi_error[2]
					self._logger.info('FN: %s (%s/%s)'  %(self.ferr[i], i, niter))
					print("Error " + 'FN: %s (%s/%s)'  %(self.ferr[i], i, niter))
				else:
					self._logger.info('Iteration: (%s/%s)'  %(i, niter))

				if i > 1 and compute_err:
					if self._converged(i):
						self.ferr = self.ferr[:i+1]
						self.err1 = self.err1[:i+1]
						self.err2 = self.err2[:i+1]
						if self.global_round == True:
								self.mmse = self.mmse[:i+1]
								self.betas = self.betas[:i+1]
						break
		end_time = time.perf_counter()
		self._logger.info("Factorization time: %s minutes" %((end_time - start_time)/60))

	def print_fun(self, x, f, accepted):
		print("at minimum %.4f accepted %d" % (f, int(accepted)))
		print(x)
	
	def fn(self, x, A, b): #
		return np.linalg.norm(A.dot(x) - b, ord = 2)
	

	def _update_h(self):
		def updatesingleH(i):		
				x0 = copy.copy(self.H[:,i])
				self.method = "SLSQP"
				minout = scipy.optimize.minimize(self.fn, x0, args=(self.W, self.data[:,i]), method=self.method,bounds=self.bounds,constraints=self.cons)
				self.H[:,i] = minout.x ##
				self.H[:,i] = minout.x / np.sum(minout.x)

		for i in range(self._num_samples):
				updatesingleH(i)
				
	def _update_w(self):

		def _jaxcosim(x, i1, i2):
			x1 = jnp.array(x[:,i1])
			x2 = jnp.array(x[:,i2])
			return jax.numpy.dot(x1, x2)/(jax.numpy.linalg.norm(x1)*jax.numpy.linalg.norm(x2))

		def calculate_jaxgrad(x):
			jacob = np.zeros([x.shape[0], x.shape[1]])
			pairs = [p for p in itertools.combinations(range(x.shape[1]),2)]
			for p in pairs:
				jacob += grad(_jaxcosim)(x, p[0], p[1])
			return jacob

		if (self.l1 != 1):
				#calculate gradient of kurtosis
				print("Calculates gradient of Cosine(W)..")
				cosine_g = calculate_jaxgrad(self.W)
				print("L1 = ", self.l1)
				print("L2 = ", self.l2)
		
				rate = np.divide(self.W, (self.l1*2*np.matmul(self.W, np.matmul(self.H, self.H.T))) + self.scaler*self.l2*cosine_g)
				self.rate = rate
				part2 = self.l1*(-2)*np.matmul(self.data, self.H.T) + 2*self.l1*np.matmul(self.W, np.matmul(self.H, self.H.T)) + self.scaler*self.l2*cosine_g
				W__ = np.multiply(self.W, np.divide( 2*self.l1*np.matmul(self.data, self.H.T), ((self.l1*2*np.matmul(self.W, np.matmul(self.H, self.H.T))) + self.scaler*self.l2*cosine_g))) 
				#print("negative vals in W__:")
				#print(np.sum(W__<0))
				
				#print("negative vals in lr:")
				#print(np.sum(rate<0))
				if (np.sum(W__==0)>0):
					W__ += 1e-10 #add pseudocount to avoid division by 0 in self.rate calculation in next iteration
				#print(W_)
				self.W = W__
		else:
				print("Initial round")
				rate = np.divide(self.W, (self.l1*2*np.matmul(self.W, np.matmul(self.H, self.H.T))))
				self.rate = rate
				part2 = self.l1*(-2)*np.matmul(self.data, self.H.T) + 2*self.l1*np.matmul(self.W, np.matmul(self.H, self.H.T))
				W__ = np.multiply(self.W, np.divide(np.matmul(self.data, self.H.T), np.matmul(self.W, np.matmul(self.H, self.H.T))))
				#print("negative vals in W__:")
				#print(np.sum(W__<0))
				if (np.sum(W__==0)>0):
					W__ += 1e-10 #add pseudocount to avoid division by 0 in self.rate calculation in next iteration
				#print(W_)
				self.W = W__

		




