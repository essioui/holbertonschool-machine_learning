#!/usr/bin/env python3
"""
Create new layer
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    Create new layer
        tf.layers.Dense: one layer
            units: numbers of nodes
            actication: activation function
            kernel_initializer: prepare of weights(He initialization)
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer(prev)
