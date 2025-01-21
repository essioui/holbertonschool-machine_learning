#!/usr/bin/env python3
"""
Define module Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    Args:
        X is the first numpy.ndarray of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features in X
        Y is the second numpy.ndarray of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features in Y
    Returns:
        the shuffled X and Y matrices
    """
    assert X.shape[0] == Y.shape[0]
    indices = np.random.permutation(X.shape[0])
    X_suffle = X[indices]
    Y_shuffle = Y[indices]
    return X_suffle, Y_shuffle
