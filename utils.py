from typing import Callable, List, Tuple
import numpy as np


def sse(preds:np.ndarray, labels:np.ndarray) -> float:
    """Computes the sum of the squared errors

    Args:
        preds (np.ndarray): predictions
        labels (np.ndarray): true values

    Returns:
        float: sum of the squared errors
    """
    return np.sum(np.inner(preds-labels, preds-labels))

def gd_optimise(params0:np.ndarray, fun:Callable, grad_fun:Callable, eta:float=1e-2, tol:float=1e-4, max_iter:int=10_000) -> Tuple[np.ndarray, List[float]]:
    fun_hist = []
    params_old = params0
    for it in range(max_iter):
        params_new = params_old - eta*grad_fun(params_old)
        fun_hist.append(fun(params_new))
        if np.all(np.abs(params_new - params_old)) < tol:
            break
        params_old = params_new
    print(f"Optimisation completed. Iterations = {it+1:d}. Final loss value = {fun_hist[-1]:4f}")
    return params_new, fun_hist

def svd_solve(A:np.ndarray, b:np.ndarray) -> np.ndarray:
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    pseudo_inv = VT.T @ np.diag(1/s) @ U.T
    return pseudo_inv @ b

def qr_solve(A:np.ndarray, b:np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(A, mode='reduced')
    y = Q.T @ b
    x = np.linalg.solve(R, y)
    return x

def compute_array_statistics(arr:np.ndarray, axis=0):
    m = arr.mean(axis=axis)
    s = arr.std(axis=axis)
    return m, s
