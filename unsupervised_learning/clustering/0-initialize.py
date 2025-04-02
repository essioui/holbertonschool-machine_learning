#!/usr/bin/env python3
"""
Module define Initialize K-means
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    Args:
        X is a numpy.ndarray of shape (n, d)
            n is the number of data points
            d is the number of dimensions
        k is a positive integer containing the number of clusters
    Returns:
        a numpy.ndarray of shape (k, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))

    return centroids
