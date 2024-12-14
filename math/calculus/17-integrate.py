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
    if all(coeff == 0 for coeff in poly):
        return [C]
    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i]
        if coeff != 0:
            integral.append(coeff / (i + 1))
    return [int(x) if
            isinstance(x, float) and x.is_integer() else x for x in integral]
