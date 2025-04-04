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

    n, d = X.shape
    k, n2 = g.shape

    if n != n2:
        return None, None, None

    Nk = np.sum(g, axis=1)
    if np.any(Nk == 0):
        return None, None, None

    m = (g @ X) / Nk[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        X_centered = X - m[i]
        gamma = g[i][:, np.newaxis]
        S[i] = (gamma * X_centered).T @ X_centered / Nk[i]

    pi = Nk / n

    return pi, m, S
