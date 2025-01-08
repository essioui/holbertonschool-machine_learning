#!/usr/bin/env python3
"""
One-Hot Decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    Args:
        one_hot is a one-hot encoded numpy.ndarray with shape (classes, m):
            classes is the maximum number of classes
            m is the number of examples
    Returns:
    numpy.ndarray with shape (m, ) or None
    """
    try:
        if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
            return None
        label = np.argmax(one_hot, axis=0)

        return label
    except Exception as e:
        return None
