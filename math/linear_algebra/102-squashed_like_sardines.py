#!/usr/bin/env python3
"""Defines cat_matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    if axis == 0:
        if isinstance(mat1[0], (int, float)) and isinstance(mat2[0], (int, float)):
            return mat1 +mat2
        if all(len(row) == len(mat2[0]) for row in mat1) and all(len(row) == len(mat1[0])for row in mat2):
            return mat1+mat2
        return None
    return None
