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
