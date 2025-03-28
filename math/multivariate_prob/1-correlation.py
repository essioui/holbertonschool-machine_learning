#!/usr/bin/env python3
"""
Module define Mean and Correlation
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    Args:
        C is a numpy.ndarray of shape (d, d):
            d is the number of dimension
    Returns:
        numpy.ndarray of shape (d, d)
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    sqrt_var = np.sqrt(np.diag(C))

    dia_matr_var = np.outer(sqrt_var, sqrt_var)

    correlation_matrix = C / dia_matr_var

    return correlation_matrix
