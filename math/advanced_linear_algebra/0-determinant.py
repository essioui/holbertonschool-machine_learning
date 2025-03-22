#!/usr/bin/env python3
"""
Module define Determinant
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    Args:
        matrix is a list of lists
    Returns:
        the determinant of matrix
    """
    if (not isinstance(matrix, list) or not
            all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    rows = len(matrix)
    if any(len(row) != rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if rows == 1:
        return matrix[0][0]

    if rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant_val = 0
    for i in range(rows):
        minor = [row[:i] + row[i+1:] for row in matrix[1:]]
        determinant_val += (-1) ** i * matrix[0][i] * determinant(minor)

    return determinant_val
