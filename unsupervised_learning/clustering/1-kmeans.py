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
        old_centroid = np.copy(C)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for i in range(k):
            cluster_points = X[clss == i]
            if cluster_points.shape[0] == 0:
                C[i] = np.random.uniform(min_vals, max_vals, (1, d))
            else:
                C[i] = np.mean(cluster_points, axis=0)

        if np.allclose(C, old_centroid):
            break

    return C, clss
