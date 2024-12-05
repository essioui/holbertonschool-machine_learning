#!/usr/bin/env python3
"""Defines cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    Args:
        mat1: 2D matrices containing ints/floats
        mat2: 2D matrices containing ints/floats
    return:
        None
        matrix
    """
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    elif axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            return mat1 + mat2
    return None
