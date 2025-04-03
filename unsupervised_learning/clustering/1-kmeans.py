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

    _, d = X.shape

    centroids = np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, d)
    )

    for i in range(iterations):
        old_centroids = np.copy(centroids)

        distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids)**2, axis=2))

        clss = np.argmin(distances, axis=1)

        for j in range(k):
            if X[clss == j].size == 0:
                centroids[j] = np.random.uniform(
                    low=np.min(X, axis=0),
                    high=np.max(X, axis=0),
                    size=(1, d)
                )
            else:
                centroids[j] = np.mean(X[clss == j], axis=0)

        if np.allclose(old_centroids, centroids):
            break

    return centroids, clss
