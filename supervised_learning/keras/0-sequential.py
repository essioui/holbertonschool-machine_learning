#!/usr/bin/env python3
"""
Module defines and builds a neural network with the Keras library.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.
    Args:
        nx: the number of input features to the network
        layers_list: a list containing the number of nodes in each layer
        activations: a list containing the activation functions
        lambtha: the L2 regularization parameter
        keep_prob: the probability that a node will be kept for dropout
    Returns:
        the keras model
    """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             kernel_regularizer=regularizer,
                             input_shape=(nx,)))

    for i in range(1, len(layers)):
        if keep_prob < 1:
            model.add(K.layers.Dropout(1 - keep_prob))

        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=regularizer))
    return model
