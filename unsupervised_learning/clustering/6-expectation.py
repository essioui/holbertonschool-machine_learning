#!/usr/bin/env python3
"""
Clustering module
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM
    Args:
        X is a numpy.ndarray of shape (n, d)
        pi is a numpy.ndarray of shape (k,)
        m is a numpy.ndarray of shape (k, d)
        S is a numpy.ndarray of shape (k, d, d)
    Returns:
        g, l, or None, None
            g is a numpy.ndarray of shape (k, n)
            l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    if (X.shape[1] != m.shape[1] or m.shape[0] != pi.shape[0]
            or S.shape[0] != pi.shape[0] or S.shape[1] != S.shape[2]
            or S.shape[1] != X.shape[1]):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    k, n = pi.shape[0], X.shape[0]

    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    likelihoods = np.sum(g, axis=0)

    if np.any(likelihoods == 0):
        return None, None

    g /= likelihoods

    log_likelihoods = np.sum(np.log(likelihoods))

    return g, log_likelihoods
