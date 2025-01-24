#!/usr/bin/env python3
"""
Defines module RMSProp
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm
    Args:
        alpha is the learning rate
        beta2 is the RMSProp weight
        epsilon is a small number to avoid division by zero
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        s is the previous second moment of var
    Returns:
        the updated variable and the new moment, respectively
    """
    S_t = beta2 * s + (1 - beta2) * (grad ** 2)
    var_update = alpha * grad / (np.sqrt(S_t) + epsilon)
    var -= var_update
    return var, S_t
