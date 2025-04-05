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

    for i in range(iterations):
        # Evaluate the probabilities and likelihoods with current parameters
        g, prev_li = expectation(X, pi, m, S)

        # In verbose mode, print the likelihood every 10 iterations after 0
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {round(prev_li, 5)}")

        # Re-estimate the parameters with the new values
        pi, m, S = maximization(X, g)

        # Evaluate new log likelihood
        g, li = expectation(X, pi, m, S)

        # If the likelihood varied by less than the tolerance value, we stop
        if np.abs(li - prev_li) <= tol:
            break

    # Last verbose message with current likelihood
    if verbose:
        # NOTE i + 1 since it has been updated once more since last print
        print(f"Log Likelihood after {i + 1} iterations: {round(li, 5)}")
    return pi, m, S, g, li
