#!/usr/bin/env python3
"""
Module define Bayesian probability
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood
    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
    Return:
        1D numpy.ndarray containing the likelihood of obtaining the data
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if (P < 0).any() or (P > 1).any():
        raise ValueError("All values in P must be in the range [0, 1]")

    comb_nx = np.math.factorial(n) // (
        np.math.factorial(x) * np.math.factorial(n - x))

    likelihoods = comb_nx * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
