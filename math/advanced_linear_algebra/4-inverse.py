#!/usr/bin/env python3
"""
Module define Inverse
"""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    Calculates the inverse of a given square matrix.
    """
    if (not isinstance(matrix, list) or not
            all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if any(len(row) != len(matrix) for row in matrix):
        raise TypeError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    if det == 0:
        return None

    adjugate_matrix = adjugate(matrix)

    inverse_matrix = ([[adjugate_matrix[i][j] / det
                        for j in range(len(adjugate_matrix))]
                       for i in range(len(adjugate_matrix))])

    return inverse_matrix
