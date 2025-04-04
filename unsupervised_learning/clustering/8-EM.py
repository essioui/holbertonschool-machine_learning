#!/usr/bin/env python3
"""
Clustering module
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    Args:
        X is a numpy.ndarray of shape (n, d)
        k is a positive integer containing the number of clusters
        iterations is a positive integer containing the maximum number
        tol is a non-negative float containing tolerance of the log likelihood
        verbose is a boolean that determines
    Returns:
        pi, m, S, g, l, or None, None, None, None, None
            pi is a numpy.ndarray of shape (k,)
            m is a numpy.ndarray of shape (k, d)
            S is a numpy.ndarray of shape (k, d, d)
            g is a numpy.ndarray of shape (k, n)
            l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    g, lkhd = expectation(X, pi, m, S)

    for i in range(iterations):
        pi, m, S = maximization(X, g)

        g, lkhd_new = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {lkhd:.5f}")

        if abs(lkhd - lkhd_new) <= 0:
            if verbose:
                print(
                    f"Log Likelihood after {i + 1} iterations: {lkhd_new:.5f}"
                )
            break

        lkhd = lkhd_new

    else:
        if verbose:
            print(f"Log Likelihood after {iterations} iterations: {lkhd:.5f}")

    return pi, m, S, g, lkhd
