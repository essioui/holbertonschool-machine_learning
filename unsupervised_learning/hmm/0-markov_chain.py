#!/usr/bin/env python3
"""
Module define Markov Models
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain
    Args:
        P is a square 2D numpy.ndarray of shape (n, n)
            P[i, j] the probability of transitioning from state i to state j
            n is the number of states in the markov chain
        s is a numpy.ndarray of shape (1, n)
        t is the number of iterations that the markov chain
    Returns:
        a numpy.ndarray of shape (1, n)
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None

    if P.shape[0] != P.shape[1] or P.shape[0] != s.shape[1]:
        return None

    results = s

    for _ in range(t):
        results = results.dot(P)

    return results
