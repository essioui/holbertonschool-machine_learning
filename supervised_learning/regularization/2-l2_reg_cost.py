#!/usr/bin/env python3
"""
Module defines L2 Regularization Cost
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization
    Args:
        cost is a tensor containing the cost without L2 regularization
        model is a Keras model that includes layers with L2 regularization
    Returns:
        tensor containing the total cost for each layer of the network,
        accounting for L2 regularization
    """
    reg_losses = [layer.losses for layer in model.layers]

    reg_losses_flattened = [loss for sublist in reg_losses for loss in sublist]

    total_cost = cost + tf.stack(reg_losses_flattened)

    return total_cost
