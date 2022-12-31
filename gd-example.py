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
	# print(f"Test objective value: {objective(params_test, X, y)}")
	# print(f"Test gradient value: {gradient(params_test, X)}")

	#%% Define the partial objective and gradient functions and instantiate some initial parameters
	f = partial(objective, X=X, y=y)
	df = partial(gradient, X=X)

	initial_params = [
		np.array([3., 2., .7, .5]) - 1.*np.random.randn(4),
		np.array([3., 2., .7, .5]) - .1*np.random.randn(4),
		np.array([3., 2., .7, .5]) - .01*np.random.randn(4),
		np.array([3., 2., .7, .5]) - .001*np.random.randn(4),
	]

	#%% Run the optimisation for all initial parameters
	histories = []

	fig, ax = plt.subplots()
	ax.plot(X_grid, generator(X_grid), 'b-', label='generator')
	ax.scatter(X, y, marker='o', facecolor='k', edgecolor='k', alpha=.7, label='data')

	for p0 in initial_params:
		p_end, hist = gd_optimise(p0, f, df, eta=.01)
		histories.append(hist)
		ax.plot(X_grid, generator(X_grid, params=p_end), label=f'p0={str(p0)}')

	ax.set(xlabel='x', ylabel='y', title='Original curve and different regressions.')
	ax.grid()
	ax.legend()
	# plt.show()

	#%% Plot the loss histories
	fig, ax = plt.subplots()
	for hist in histories:
		ax.plot(hist)
	ax.set(xlabel='iterations', ylabel='SSE', title='Sum of squared errors during optimization.')
	plt.show()
