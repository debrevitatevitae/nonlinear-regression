from functools import partial
import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import sse, gd_optimise


def generator(x:float, params:np.ndarray=np.array([.3, 2., 9.])) -> float:
	return params[0]*x**5 + params[1]*x**2 + params[2]

def generate_polynomial_feature_matrix(X:np.ndarray, p:int=10) -> np.ndarray:
	n = X.size
	A = np.empty((n, p))
	A[:, 0] = np.zeros((n,))
	for k in range(1, p):
		A[:, k] = X**k
	return A


if __name__ == '__main__':
	np.random.seed(0)

	#%% Generate some sample data and add some noise
	X = np.random.uniform(low=0., high=2., size=(100,))
	X_grid = np.linspace(0., 2., 1000)
	y = generator(X) + .5*np.random.randn(100)

	fig, ax = plt.subplots()
	ax.plot(X_grid, generator(X_grid), 'b-', label='generator')
	ax.scatter(X, y, marker='o', facecolor='k', edgecolor='k', alpha=.7, label='data')
	ax.set(xlabel='x', ylabel='y', title='Generator and data example for regression.')
	ax.grid()
	ax.legend()
	plt.show()

	#%% Generate the polynomial data matrix
	A = generate_polynomial_feature_matrix(X, p=20)

	#%% Loop for 
