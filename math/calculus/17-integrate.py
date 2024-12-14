#!/usr/bin/env python3
"""poly_integral"""


def poly_integral(poly, C=0):
    """
    integral of a polynomial
    """
    if not isinstance(poly, list) or not all(
            (isinstance(x, (int, float))for x in poly)) or not (
            isinstance(C, (int, float))):
        return None
    integral = []
    for i, coeff in enumerate(poly):
        if coeff != 0:
            integral.append(coeff / (i + 1))
    integral.insert(0, C)
    return integral
