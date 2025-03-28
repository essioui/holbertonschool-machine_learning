#!/usr/bin/env python3
"""
Module define Mean and Covariance
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance
    Args:
        n is the number of data points
        d is the number of dimensions in each data point
    Returns: mean, cov:
        mean is a numpy.ndarray of shape (1, d)
        cov is a numpy.ndarray of shape (d, d)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    X_centrer = X - mean

    covariance = X_centrer.T @ X_centrer / (n - 1)

    return mean, covariance
