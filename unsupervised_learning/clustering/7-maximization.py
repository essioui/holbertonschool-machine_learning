#!/usr/bin/env python3
"""
Clustering module
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    Args:
        X is a numpy.ndarray of shape (n, d)
        g is a numpy.ndarray of shape (k, n)
    Returns:
        pi, m, S, or None, None, None
            pi is a numpy.ndarray of shape (k,)
            m is a numpy.ndarray of shape (k, d)
            S is a numpy.ndarray of shape (k, d, d)
    """
    if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
        return None, None, None

    if len(X.shape) != 2 or len(g.shape) != 2:
        return None, None, None

    if X.shape[0] != g.shape[1] or not np.allclose(g.sum(axis=0), 1.0):
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    # Update the priors
    pi = np.sum(g, axis=1) / n

    # Update the centroids
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    # Update the covariance matrices, using the new centroids
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
