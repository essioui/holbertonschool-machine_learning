#!/usr/bin/env python3
"""
Module define Entropy
"""
import numpy as np


def HP(Di: np.ndarray, beta: np.ndarray):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    Args:
        Di is a numpy.ndarray of shape (n - 1,)
            n is the number of data point
        beta is a numpy.ndarray of shape (1,)
    Returns: (Hi, Pi):
        Hi: the Shannon entropy of the points
        Pi: a numpy.ndarray of shape (n - 1,)
    """
    Pi = np.exp(-Di * beta)

    sum_Pi = np.sum(Pi)

    Pi /= sum_Pi

    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
