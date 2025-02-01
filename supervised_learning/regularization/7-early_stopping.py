#!/usr/bin/env python3
"""
Module defines Early Stopping
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early
    Args:
        cost is the current validation cost of the neural network
        opt_cost is the lowest recorded validation cost
        threshold is the threshold used for early stopping
        patience is the patience count used for early stopping
        count is the count of how long the threshold has not been met
    Returns:
        boolean of whether the network should be stopped early,
        followed by the updated count
    """
    if cost < opt_cost - threshold:
        return False, 0
    elif count + 1 >= patience:
        return True, count + 1
    else:
        return False, count + 1
