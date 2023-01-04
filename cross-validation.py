from functools import partial
import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from utils import svd_solve, compute_array_statistics


def generator(x:float, params:np.ndarray=np.array([1.])) -> float:
	return params[0]*x**2

def generate_polynomial_feature_matrix(X:np.ndarray, p:int=20) -> np.ndarray:
	n = X.size
	A = np.empty((n, p))
	A[:, 0] = np.ones((n,))
	for k in range(1, p):
		A[:, k] = X**k
	return A


if __name__ == '__main__':
	np.random.seed(0)

	#%% Generate some sample data and add some noise
	n = 1000
	X = np.random.uniform(0., 4., n)
	y = generator(X) + 0.1*np.random.randn(n)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
	n_train = len(y_train)
	max_degree = 20
	A = generate_polynomial_feature_matrix(X_train, p=max_degree)

	#%% (Test) Solve a linear system with the three methods we'll be using
	x_lstsq = np.linalg.lstsq(A, y_train, rcond=None)[0]
	x_pinv = svd_solve(A, y_train)
	lasso = linear_model.Lasso(alpha=.1, fit_intercept=False)
	lasso.fit(A, y_train)
	x_lasso = lasso.coef_


	fig, axs = plt.subplots(1, 3)
	axs[0].bar(range(max_degree), x_lstsq)
	axs[1].bar(range(max_degree), x_pinv)
	axs[2].bar(range(max_degree), x_lasso)
	# plt.show()

	#%% Train with different methods for different folds
	Ks = [2, 10, 20]
	
	fig_coefs, axs_coefs = plt.subplots(3, 3)
	fig_err, axs_err = plt.subplots(1, 3)

	for j, K in enumerate(Ks):
		coefs_lstsq = np.empty((K, max_degree))
		coefs_pinv = np.empty((K, max_degree))
		coefs_lasso = np.empty((K, max_degree))
		E_lstsq = np.empty((K,))
		E_pinv = np.empty((K,))
		E_lasso = np.empty((K,))

		for i in range(K):
			# select K samples
			idxs = np.random.randint(0, n_train, size=(n_train//K,))
			X_i = X_train[idxs]
			y_i = y_train[idxs]
			# Genrate the feature matrix
			A_i = generate_polynomial_feature_matrix(X_i, p=max_degree)
			# Perform regression with different methods
			c_lstsq = np.linalg.lstsq(A_i, y_i, rcond=None)[0]
			c_pinv = svd_solve(A_i, y_i)
			lasso.fit(A_i, y_i)
			c_lasso = lasso.coef_
			# Compute the relative errors
			e_lstsq = np.linalg.norm(A_i @ c_lstsq - y_i) / np.linalg.norm(y_i)
			e_pinv = np.linalg.norm(A_i @ c_pinv - y_i) / np.linalg.norm(y_i)
			e_lasso = np.linalg.norm(A_i @ c_lasso - y_i) / np.linalg.norm(y_i)
			# Store everything
			coefs_lstsq[i, :] = c_lstsq
			coefs_pinv[i, :] = c_pinv
			coefs_lasso[i, :] = c_lasso
			E_lstsq[i] = e_lstsq
			E_pinv[i] = e_pinv
			E_lasso[i] = e_lasso

		# Compute the average coefficients as final and the average training errors
		c_ave_lstsq = coefs_lstsq.mean(axis=0)
		c_ave_pinv = coefs_pinv.mean(axis=0)
		c_ave_lasso = coefs_lasso.mean(axis=0)

		e_ave_lstsq = E_lstsq.mean()
		e_ave_pinv = E_pinv.mean()
		e_ave_lasso = E_lasso.mean()

		# Plot the final coefficients for this K and the error
		axs_coefs[0, j].bar(range(max_degree), c_ave_lstsq)
		axs_coefs[1, j].bar(range(max_degree), c_ave_pinv)
		axs_coefs[2, j].bar(range(max_degree), c_ave_lasso)

		axs_err[j].bar(range(3), [e_ave_lstsq, e_ave_pinv, e_ave_lasso])

	plt.show()
