#!/usr/bin/env python3
"""
Module define Definiteness
"""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix.
    Returns one of the following strings:
    - 'Positive definite'
    - 'Positive semi-definite'
    - 'Negative semi-definite'
    - 'Negative definite'
    - 'Indefinite'
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return 'Positive definite'
    elif np.all(eigenvalues >= 0):
        return 'Positive semi-definite'
    elif np.all(eigenvalues < 0):
        return 'Negative definite'
    elif np.all(eigenvalues <= 0):
        return 'Negative semi-definite'
    else:
        return None
