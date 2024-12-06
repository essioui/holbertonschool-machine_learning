#!/usr/bin/env python3
"""Defines np_cat"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    Args:
        mat1: numpy.ndarray
        mat2: numpy.ndarray
    Return:
        matrix concatenated two matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
