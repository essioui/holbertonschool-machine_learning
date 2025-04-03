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

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        if np.array_equal(clss, new_clss):
            break

        clss = new_clss

        for i in range(k):
            clusters_points = X[clss == i]

            if clusters_points.shape[0] == 0:
                C[i] = np.random.uniform(min_vals, max_vals, (1, d))

            else:
                C[i] = np.mean(clusters_points, axis=0)

    return C.copy(), clss.copy()
