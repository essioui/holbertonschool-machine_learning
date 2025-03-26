#!/usr/bin/env python3
"""
Module define Bayesian probability
"""
from scipy.special import comb


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
            "x must be an integerthat is greater than or equal to 0"
            )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if (P < 0).any() or (P > 1).any():
        raise ValueError("All values in P must be in the range [0, 1]")

    likehoods = comb(n, x) * (P ** x) * ((1 - P) ** (n - x))

    return likehoods
