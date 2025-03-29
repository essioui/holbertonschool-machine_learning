#!/usr/bin/env python3
"""
Module define Dimensionality Reduction
"""
import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on the dataset X and return the weights matrix W
    that maintains the specified fraction of the variance.
    
    Parameters:
    X: numpy.ndarray of shape (n, d), the dataset to perform PCA on
    var: float, the fraction of variance to retain (default is 0.95)
    
    Returns:
    W: numpy.ndarray of shape (d, nd), the weights matrix
    """
    U, s, V = np.linalg.svd(X)

    cumulated = np.cumsum(s)

    percentage = cumulated / np.sum(s)

    r = np.argwhere(percentage >= var)[0, 0]

    return V[:r + 1].T
