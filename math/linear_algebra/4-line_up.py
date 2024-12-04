#!/usr/bin/env python3
"""Defines add_arrays"""


def add_arrays(arr1, arr2):
    """
    Add element by element in 2 arrays (we can use zip())
    Args:
        arr1: list of ints/floats
        arr2: list of ints/floats
    Return:
        List
        None
    """
    if len(arr1) != len(arr2):
        return None

    result = []

    for i in range(len(arr1)):
        result.append(arr1[i]+arr2[i])
    return result
