#!/usr/bin/env python3
"""
Module define Markov Models
"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain
    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n)
            P[i, j] the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns:
        a numpy.ndarray of shape (1, n)
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    n, m = P.shape

    if n != m:
        return None

    pk = np.linalg.matrix_power(P, 100)

    if not np.all(pk > 0):
        return None

    A = P.T - np.eye(n)

    A = np.vstack([A, np.ones(n)])

    b = np.zeros(n + 1)

    b[-1] = 1

    try:
        steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
        return steady_state.reshape(1, n)

    except Exception:
        return None
