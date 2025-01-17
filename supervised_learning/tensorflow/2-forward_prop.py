#!/usr/bin/env python3
"""
Forward Propagation
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    Args:
        x is the placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
    """
    if len(layer_sizes) != len(activations):
        raise ValueError("must be length equals")
    output = x
    for i in range(len(layer_sizes)):
        output = create_layer(output, layer_sizes[i], activations[i])
    return output
