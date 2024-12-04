#!/usr/bin/env python3
"""Defines matrix_shape"""


def matrix_shape(matrix):
    """
    Returns the dimensions of a matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
