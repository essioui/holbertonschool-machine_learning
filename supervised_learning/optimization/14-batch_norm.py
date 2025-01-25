#!/usr/bin/env python3
"""
Defines module Batch Normalization Upgraded
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    Args:
        - prev is the activated output of the previous layer
        - n is the number of nodes in the layer to be created
        - activation is the activation function that should be
          used on the output of the layer
    Returns:
        a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(units=n, activation=None,
                                        kernel_initializer=init)

    x_prev = dense_layer(prev)

    mean, variance = tf.nn.moments(x_prev, axes=[0])

    gamma = tf.Variable(initial_value=tf.ones((1, n)),
                        trainable=True, name='gamma')
    beta = tf.Variable(initial_value=tf.zeros((1, n)),
                       trainable=True, name='beta')

    epsilon = 1e-7

    batch_norm = tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )

    return activation(batch_norm)
