#!/usr/bin/env python3
"""Defines cat_matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    if axis == 0:
        # concatenate 1D lists
        if isinstance(mat1[0], (int, float)) and \
              isinstance(mat2[0], (int, float)):
            return mat1 + mat2
        # concatenate 2D lists
        if (all(len(row) == len(mat2[0]) for row in mat1) and
                all(len(row) == len(mat1[0])for row in mat2)):
            return mat1+mat2
        return None
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    return None
