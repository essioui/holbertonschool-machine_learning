#!/usr/bin/env python3
"""
Module define Cofactor
"""


def determinant(matrix):
    """
    calculate the determinant
    """
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)

    return det


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    """
    if (not isinstance(matrix, list) or not
            all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a list of lists")

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)
    cofactor_matrix = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):

            sub_matrix = ([row[:j] + row[j+1:] for k, row
                           in enumerate(matrix) if k != i])

            minor = determinant(sub_matrix)

            cofactor_matrix[i][j] = ((-1) ** (i + j)) * minor

    return cofactor_matrix
