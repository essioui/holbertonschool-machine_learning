#!/usr/bin/env python3
"""
Module define  PCA v2
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on dataset
    Args:
        X (np.ndarray): of shape (n, d) where n is the number of data points
            ans d is the number of dimensions.
        ndims (float): is the dimensionality of the transformed X.
    Retruns:
        np.ndarray: the transformed X of shape (n, ndim).
    """
    X_m = X - np.mean(X, axis=0)

    U, s, V = np.linalg.svd(X_m)

    W = V[:ndim].T

    return np.matmul(X_m, W)
