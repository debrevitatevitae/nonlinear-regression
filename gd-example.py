import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import sse


def generator(x:float, params:np.ndarray=np.array([3., 2., .7, .5])) -> float:
	return params[0]*np.cos(params[1]*x + params[2]) + params[3]

def generate_data(n:int, noise_mag:float=0.01, params:np.ndarray=np.array([3., 2., .7, .5])) -> Tuple[np.ndarray, np.ndarray]:
	X = np.random.uniform(-np.pi/2, np.pi/2, size=(n,))
	y = generator(X, params=params) + noise_mag*np.random.randn(n)
	return X, y

def objective(params:np.ndarray, X:np.ndarray, y:np.ndarray) -> float:
	preds = generator(X, params=params)
	return sse(preds, y)

def gradient(params:np.ndarray, X:np.ndarray):
	grad = np.array(
		[
		2*np.sum((params[0]*np.cos(params[1]*X + params[2]) + params[3] - y) * (np.cos(params[1]*X + params[2]))),
		2*np.sum((params[0]*np.cos(params[1]*X + params[2]) + params[3] - y) * (-np.sin(params[1]*X + params[2])) * X),
		2*np.sum((params[0]*np.cos(params[1]*X + params[2]) + params[3] - y) * (-np.sin(params[1]*X + params[2]))),
		2*np.sum(params[0]*np.cos(params[1]*X + params[2]) + params[3] - y)
		]
	)
	return grad


if __name__ == '__main__':
	np.random.seed(0)
	
	#%% Generate data and plot generator and data
	X_grid = np.linspace(-np.pi/2, np.pi/2, 200)
	X, y = generate_data(20, noise_mag=.1)

	fig, ax = plt.subplots()
	ax.plot(X_grid, generator(X_grid), 'b-', label='generator')
	ax.scatter(X, y, marker='o', facecolor='k', edgecolor='k', alpha=.7, label='data')
	ax.set(xlabel='x', ylabel='y', title='Generator and data for regression.')
	ax.grid()
	ax.legend()
	# plt.show()

	#%% Test the objective function and the gradient function
	params_test = np.random.uniform(low=0., high=5., size=(4,))
	print(f"Test objective value: {objective(params_test, X, y)}")
	print(f"Test gradient value: {gradient(params_test, X)}")