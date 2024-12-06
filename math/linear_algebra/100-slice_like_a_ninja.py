#!/usr/bin/env python3
"""Defines np_slice"""


def np_slice(matrix, axes={}):
    """
    slices a matrix along specific axes
    Args:
        matrix: numpy.ndarray
        axex: tuple have params of slice(start, stop, step)
    Return:
        matrix
    """
    slices = []
    for axis in range(matrix.ndim):
        slice_params = axes.get(axis, (None, None))
        slices.append(slice(*slice_params))
    return matrix[tuple(slices)]
