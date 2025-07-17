#!/usr/bin/env python3
"""
Simple Policy function
"""
import numpy as np


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def policy(state, weight):
    """
    Compute the policy for a given state and weight.
    Args:
        state: The input state.
        weight: The weights of the policy.
    Returns:
        The action probabilities.
    """
    z = np.matmul(state, weight)
    return softmax(z)
