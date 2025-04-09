#!/usr/bin/env python3
"""
Module define Markov Models
"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n)
            P[i, j] is the probability of transitioning
            n is the number of states in the markov chain
    Returns:
        True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    n, m = P.shape

    if n != m:
        return None

    for i in range(n):
        if P[i, i] == 1 and (np.all(P[i, :] == 0) or np.sum(P[i, :]) == 1):
            return True

    return False
