#!/usr/bin/env python3
"""Defines cat_matrices"""
import numpy as np


def cat_matrices(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    try:
        # transfer lists to arrays numpy for use np.conacatenate
        np_mat1 = np.array(mat1)
        np_mat2 = np.array(mat2)
        result = np.concatenate((np_mat1, np_mat2), axis=axis)
        # return the result as lists
        return result.tolist()
    except ValueError:
        return None
