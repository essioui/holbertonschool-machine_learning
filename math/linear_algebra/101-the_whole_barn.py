#!/usr/bin/env python3
"""Defines add_matrices"""


def add_matrices(mat1, mat2):
    """
    Add two matrices have the same shape
    """
    if len(mat1) != len(mat2):
        return None

    def add_elements(a, b):
        """
        add elements from list
        """
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return None
            result = [add_elements(x, y) for x, y in zip(a, b)]
            if None in result:
                return None
            return result
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a + b
        else:
            return None

    return add_elements(mat1, mat2)
