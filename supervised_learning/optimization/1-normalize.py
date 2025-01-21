#!/usr/bin/env python3
"""
Defines module Normalize
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    Args:
        X is the numpy.ndarray of shape (d, nx) to normalize
            d is the number of data points
            nx is the number of features
        m is a numpy.ndarray of shape (nx,) contains the mean of X
        s is a numpy.ndarray of shape (nx,) contains the features of X
    Returns:
        The normalized X matrix
    """
    return (X - m) / s
