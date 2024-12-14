#!/usr/bin/env python3
"""summation_i_squared(n)"""


def summation_i_squared(n):
    """
    sum of number from 1 to n square
    """
    if type(n) is not int:
        return None
    elif n < 1:
        return None
    else:
        return int((n * (n + 1) * ((2 * n) + 1)) / 6)
