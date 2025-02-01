#!/usr/bin/env python3
"""
Module defines Create a Layer with L2 Regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in tensorFlow
    that includes L2 regularization
    Args:
        prev is a tensor containing the output of the previous laye
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        lambtha is the L2 regularization parameter
    Returns:
        the output of the new layer
    """
    regularizer = tf.keras.regularizers.l2(lambtha)

    tensor = tf.keras.layers.Dense(units=n, activation=activation,
                                   kernel_regularizer=regularizer)

    return tensor(prev)
