from functools import partial
import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

from utils import svd_solve, qr_solve, compute_array_statistics


def generator(x:float, params:np.ndarray=np.array([1., 2., 3.])) -> float:
	return params[0]*x**3 + params[1]*x**2 + params[2]

def generate_polynomial_feature_matrix(X:np.ndarray, p:int=10) -> np.ndarray:
	n = X.size
	A = np.empty((n, p))
	A[:, 0] = np.ones((n,))
	for k in range(1, p):
		A[:, k] = X**k
	return A

def regress_multi_methods(A:np.ndarray, b:np.ndarray) -> Tuple[np.ndarray, np.ndarray,
						np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	# least squares solution
	x_ls = svd_solve(A, b)
	# qr decomposition solution
	x_qr = qr_solve(A, b)
	# lasso with l1 penalty = 1.
	lasso_1 = linear_model.Lasso(alpha=1., fit_intercept=False)
	lasso_1.fit(A, b)
	x_lasso_1 = lasso_1.coef_
	# lasso with l1 penalty = .5
	lasso_2 = linear_model.Lasso(alpha=.5, fit_intercept=False)
	lasso_2.fit(A, b)
	x_lasso_2 = lasso_2.coef_
	# ridge regression
	ridge = linear_model.Ridge(alpha=1.)
	ridge.fit(A, b)
	x_ridge = ridge.coef_
	# huber-penalty regression
	huber = linear_model.HuberRegressor()
	huber.fit(A, b)
	x_huber = huber.coef_
	# output
	return x_ls, x_qr, x_lasso_1, x_lasso_2, x_ridge, x_huber


if __name__ == '__main__':
	np.random.seed(0)

	#%% Generate some sample data and add some noise
	X = np.random.uniform(0., 4., 100)
	X_grid = np.linspace(0., 4., 1000)
	y = generator(X)

	fig, ax = plt.subplots()
	ax.plot(X_grid, generator(X_grid), 'b-', label='generator')
	ax.scatter(X, y + .5*np.random.randn(100), marker='o', facecolor='k', edgecolor='k', alpha=.7, label='data')
	ax.set(xlabel='x', ylabel='y', title='Generator and data example for regression.')
	ax.grid()
	ax.legend()
	plt.show()

	#%% Generate the polynomial data matrix
	A = generate_polynomial_feature_matrix(X, p=5)
	print(A)
	# sys.exit()

	#%% Loop for new random noise on the data and store coefficients
	coeffs_ls = []
	coeffs_qr = []
	coeffs_lasso_1 = []
	coeffs_lasso_2 = []
	coeffs_ridge = []
	coeffs_huber = []

	for _ in range(100):
		b = y + .1*np.random.randn(100)
		c_ls, c_qr, c_lasso_1, c_lasso_2, c_ridge, c_huber = regress_multi_methods(A, b)
		coeffs_ls.append(c_ls)
		coeffs_qr.append(c_qr)
		coeffs_lasso_1.append(c_lasso_1)
		coeffs_lasso_2.append(c_lasso_2)
		coeffs_ridge.append(c_ridge)
		coeffs_huber.append(c_huber)

	# print(coeffs_ls[-1])
	# print(coeffs_qr[-1])
	# print(coeffs_lasso_1[-1])
	# print(coeffs_lasso_2[-1])
	# print(coeffs_ridge[-1])
	# print(coeffs_huber[-1])

	#%% Compute mean and standard dev for each method
	mu_ls, sig_ls = compute_array_statistics(np.array(coeffs_ls), axis=0)
	mu_qr, sig_qr = compute_array_statistics(np.array(coeffs_qr), axis=0)
	mu_lasso_1, sig_lasso_1 = compute_array_statistics(np.array(coeffs_lasso_1), axis=0)
	mu_lasso_2, sig_lasso_2 = compute_array_statistics(np.array(coeffs_lasso_2), axis=0)
	mu_ridge, sig_ridge = compute_array_statistics(np.array(coeffs_ridge), axis=0)
	mu_huber, sig_huber = compute_array_statistics(np.array(coeffs_huber), axis=0)

	#%% Plot mean and standard deviations on a bar plot for each method
	fig, axs = plt.subplots(2, 3)
	axs[0, 0].errorbar(np.arange(5), mu_ls, yerr=sig_ls, fmt='.', ecolor='red', capsize=3., capthick=1.)
	axs[0, 0].set_title('LS')
	axs[0, 1].errorbar(np.arange(5), mu_qr, yerr=sig_qr, fmt='.', ecolor='red', capsize=3., capthick=1.)
	axs[0, 1].set_title('QR')
	axs[0, 2].errorbar(np.arange(5), mu_lasso_1, yerr=sig_lasso_1, fmt='.', ecolor='red', capsize=3., capthick=1.)
	axs[0, 2].set_title('Lasso, alpha=1')
	axs[1, 0].errorbar(np.arange(5), mu_lasso_2, yerr=sig_lasso_2, fmt='.', ecolor='red', capsize=3., capthick=1.)
	axs[1, 0].set_title('Lasso, alpha=0.5')
	axs[1, 1].errorbar(np.arange(5), mu_ridge, yerr=sig_ridge, fmt='.', ecolor='red', capsize=3., capthick=1.)
	axs[1, 1].set_title('Ridge')
	axs[1, 2].errorbar(np.arange(5), mu_huber, yerr=sig_huber, fmt='.', ecolor='red', capsize=3., capthick=1.)
	axs[1, 2].set_title('Huber')
	fig.suptitle('Coefficients for different solvers')
	plt.show()