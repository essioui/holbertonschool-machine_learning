#!/usr/bin/env python3
"""Defines mat_mul"""


def mat_mul(mat1, mat2):
    """
    performs matrix multiplication
    Args:
        mat1: 2D matrices containing ints/floats
        mat2: 2D matrices containing ints/floats
    Return:
        None or matrix
    """
    if len(mat1[0]) != len(mat2):
        return None
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for l in range(len(mat2)):
                result[i][j] += mat1[i][l]*mat2[l][j]
    return result
