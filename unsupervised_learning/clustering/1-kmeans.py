#!/usr/bin/env python3
"""
Module define K-means
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    Args:
        X is a numpy.ndarray of shape (n, d)
            n is the number of data points
            d is the number of dimensions
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing the maximum of iterations
    Returns:
        C, clss, or None, None on failure
            C is a numpy.ndarray of shape (k, d)
            clss is a numpy.ndarray of shape (n,)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    C = np.random.uniform(min_vals, max_vals, (k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        Centroid = np.copy(C)

        for i in range(k):
            if np.sum(clss == i) == 0:
                Centroid[i] = np.random.uniform(min_vals, max_vals, (1, d))
            else:
                Centroid[i] = np.mean(X[clss == i], axis=0)

        if np.array_equal(Centroid, C):
            return C, clss
        else:
            C = Centroid

    return C.copy(), clss.copy()
