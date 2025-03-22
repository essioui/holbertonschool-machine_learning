#!/usr/bin/env python3
"""
Module define Minor
"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    """
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise ValueError("matrix must be a list of lists")

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)
    if size == 1:
        return [[1]]

    minor_matrix = []
    for i in range(size):
        row_minors = []
        for j in range(size):
            minor = ([row[:j] + row[j+1:] for
                      row in (matrix[:i] + matrix[i+1:])])
            row_minors.append(determinant(minor))
        minor_matrix.append(row_minors)

    return minor_matrix
