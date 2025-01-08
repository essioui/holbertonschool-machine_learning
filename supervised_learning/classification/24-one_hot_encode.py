#!/usr/bin/env python3
"""
One-Hot Encode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    Args:
        Y is a numpy.ndarray with shape (m,):
            m is the number of examples
        classes is the maximum number of classes found in Y
    Returns:
        one-hot encoding of Y with shape (classes, m), or None
    """
    try:
        if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
            return None
        if Y.ndim != 1 or classes <= np.max(Y):
            return None
        one_hot = np.zeros((classes, Y.shape[0]))

        one_hot[Y, np.arange(Y.shape[0])] = 1

        return one_hot
    except Exception as e:
        return None
