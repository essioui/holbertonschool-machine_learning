#!/usr/bin/env python3
"""
Module define Adjugate
"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    """
    cofactor_matrix = cofactor(matrix)

    adjugate_matrix = ([[cofactor_matrix[j][i] for j in range(
        len(cofactor_matrix))] for i in range(len(cofactor_matrix))])

    return adjugate_matrix
