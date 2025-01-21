#!/usr/bin/env python3
"""
Defines module Normalization Constants
"""
import numpy as np


def normalization_constants(X):
    """
    calculates the normalization (standardization) constants of a matrix
    Args:
        X is the numpy.ndarray of shape (m, nx) to normalize
            m is the number of data points
            nx is the number of features
    Returns:
        the mean and standard deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)

    return mean, std_dev
