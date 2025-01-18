#!/usr/bin/env python3
"""
Module defines and builds a neural network with the Keras library.
Will build model by  Functional API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library (Functional API)
    Args:
        nx is the number of input features to the network
        layers is a list containing the number of nodes in each layer
        activations is a list containing the activation functions
        lambtha is the L2 regularization parameter
        keep_prob is the probability that a node will be kept for dropout
    Returns:
        the keras model
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)

    layer = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=regularizer)(inputs)

    for i in range(1, len(layers)):
        layer = K.layers.Dropout(1 - keep_prob)(layer)
        layer = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=regularizer)(layer)

    return K.Model(inputs=inputs, outputs=layer)
