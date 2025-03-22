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
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix) >= 1 and np.array_equal(matrix, matrix.T):
        eigvals = np.linalg.eigvals(matrix)
        '''print(matrix)
        print("eigvals", eigvals)'''
        if np.all(eigvals > 0):
            return "Positive definite"
        elif np.all(eigvals >= 0):
            return "Positive semi-definite"
        elif np.all(eigvals < 0):
            return "Negative definite"
        elif np.all(eigvals <= 0):
            return "Negative semi-definite"
        else:
            return "Indefinite"
    else:
        return None
