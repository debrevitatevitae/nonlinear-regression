from functools import partial
import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
	
	#%% Scale features
	scaler = StandardScaler()
	A = scaler.fit_transform(A)

	#%% (Test) Solve a linear system with the three methods we'll be using
	x_lstsq = np.linalg.lstsq(A, y_train, rcond=None)[0]
	x_pinv = svd_solve(A, y_train)
	lasso = linear_model.Lasso(alpha=.1, fit_intercept=True, max_iter=1000)
	lasso.fit(A, y_train)
	x_lasso = lasso.coef_


	fig, axs = plt.subplots(1, 3)
	axs[0].bar(range(max_degree), x_lstsq)
	axs[1].bar(range(max_degree), x_pinv)
	axs[2].bar(range(max_degree), x_lasso)
	axs[0].set_title('LSTSQ')
	axs[1].set_title('Pinv')
	axs[2].set_title('LASSO')
	fig.supxlabel('polynomial degree')
	fig.supylabel('weights')
	plt.show(block=False)
	# plt.close(fig)
	# sys.exit()

	#%% Train with different methods for different folds
	Ks = [2, 10, 100]
	
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
			A_i = scaler.transform(A_i)
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
		axs_err[j].set_xticks(range(3))
		axs_err[j].set_xticklabels(['LSTSQ', 'Pinv', 'LASSO'], rotation=45)
		axs_err[j].set_title(f'K={K:d}')

	axs_coefs[0, 0].set_title(f'K={Ks[0]:d}')
	axs_coefs[0, 0].set_ylabel('LSTSQ')
	axs_coefs[0, 1].set_title(f'K={Ks[1]:d}')
	axs_coefs[0, 2].set_title(f'K={Ks[2]:d}')
	axs_coefs[1, 0].set_ylabel('Pinv')
	axs_coefs[2, 0].set_ylabel('LASSO')
	fig_coefs.supxlabel('polynomial degree')
	fig_coefs.suptitle('Weights for different methods and different CV folds')
	fig_coefs.tight_layout()

	axs_err[0].set_ylabel('train error')
	fig_err.suptitle('Training error for different methods and different CV folds')
	fig_err.tight_layout()

	plt.show(block=False)

	#%% Compute the test errors for K=100 and compare against the averages of the 100 train errors
	A_test = generate_polynomial_feature_matrix(X_test, p=max_degree)
	A_test = scaler.transform(A_test)
	e_test_lstsq = np.linalg.norm(A_test @ c_ave_lstsq - y_test) / np.linalg.norm(y_test)
	e_test_pinv = np.linalg.norm(A_test @ c_ave_pinv - y_test) / np.linalg.norm(y_test)
	e_test_lasso = np.linalg.norm(A_test @ c_ave_lasso - y_test) / np.linalg.norm(y_test)

	fig1, axs1 = plt.subplots(1, 3)
	axs1[0].bar(range(3), [e_ave_lstsq, e_ave_pinv, e_ave_lasso])
	axs1[0].set_xticks(range(3))
	axs1[0].set_xticklabels(['LSTSQ', 'Pinv', 'LASSO'], rotation=45)
	axs1[0].set_ylabel('Error')
	axs1[0].set_title('Train error')
	axs1[1].bar(range(3), [e_test_lstsq, e_test_pinv, e_test_lasso])
	axs1[1].set_xticks(range(3))
	axs1[1].set_xticklabels(['LSTSQ', 'Pinv', 'LASSO'], rotation=45)
	axs1[1].set_title('Test error')
	axs1[2].bar(range(3), [e_test_lstsq, e_test_pinv, e_test_lasso])
	axs1[2].set_xticks(range(3))
	axs1[2].set_xticklabels(['LSTSQ', 'Pinv', 'LASSO'], rotation=45)
	axs1[2].set_title('Test error (zoom)')
	axs1[2].set(ylim = [0, 10])
	fig1.suptitle('Train and test error')
	fig1.tight_layout()
	plt.show(block=False)

	#%% Add thresholding of the coefficients and replot training and test error
	# This time, the training error is computed over the entire dataset rather than being averaged
	# over the k-folds
	th_lstsq = .1 * np.max(np.abs(c_ave_lstsq))
	th_pinv = .1 * np.max(np.abs(c_ave_pinv))
	th_lasso = .1 * np.max(np.abs(c_ave_lasso))

	c_th_lstsq = np.where(np.abs(c_ave_lstsq) > th_lstsq, c_ave_lstsq, np.zeros(max_degree))
	c_th_pinv = np.where(np.abs(c_ave_pinv) > th_pinv, c_ave_pinv, np.zeros(max_degree))
	c_th_lasso = np.where(np.abs(c_ave_lasso) > th_lasso, c_ave_lasso, np.zeros(max_degree))

	print(c_th_lasso)

	e_th_train_lstsq = np.linalg.norm(A @ c_th_lstsq - y_train) / np.linalg.norm(y_train)
	e_th_train_pinv = np.linalg.norm(A @ c_th_pinv - y_train) / np.linalg.norm(y_train)
	e_th_train_lasso = np.linalg.norm(A @ c_th_lasso - y_train) / np.linalg.norm(y_train)

	e_th_test_lstsq = np.linalg.norm(A_test @ c_th_lstsq - y_test) / np.linalg.norm(y_test)
	e_th_test_pinv = np.linalg.norm(A_test @ c_th_pinv - y_test) / np.linalg.norm(y_test)
	e_th_test_lasso = np.linalg.norm(A_test @ c_th_lasso - y_test) / np.linalg.norm(y_test)

	fig2, axs2 = plt.subplots(1, 3)
	axs2[0].bar(range(3), [e_th_train_lstsq, e_th_train_pinv, e_th_train_lasso])
	axs2[0].set_xticks(range(3))
	axs2[0].set_xticklabels(['LSTSQ', 'Pinv', 'LASSO'], rotation=45)
	axs2[0].set_ylabel('Error')
	axs2[0].set_title('Train error')
	axs2[1].bar(range(3), [e_th_test_lstsq, e_th_test_pinv, e_th_test_lasso])
	axs2[1].set_xticks(range(3))
	axs2[1].set_xticklabels(['LSTSQ', 'Pinv', 'LASSO'], rotation=45)
	axs2[1].set_ylabel('Error')
	axs2[1].set_title('Test error')
	axs2[2].bar(range(3), [e_th_test_lstsq, e_th_test_pinv, e_th_test_lasso])
	axs2[2].set_xticks(range(3))
	axs2[2].set_xticklabels(['LSTSQ', 'Pinv', 'LASSO'], rotation=45)
	axs2[2].set_title('Test error (zoom)')
	axs2[2].set(ylim = [0, 10])
	fig2.suptitle('Train and test error with weights truncation')
	fig2.tight_layout()
	plt.show()