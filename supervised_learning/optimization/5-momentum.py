#!/usr/bin/env python3
"""
Defines module Momentum
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
     Updates a variable using the gradient descent
     with momentum optimization algorithm
     Args:
        alpha is the learning rate
        beta1 is the momentum weight
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        v is the previous first moment of var
    Returns:
        the updated variable and the new moment, respectively
    """
    update_variable = (beta1 * v) + ((1 - beta1) * grad)
    new_moment = var - (alpha * update_variable)
    return new_moment, update_variable
