#!/usr/bin/env python3
"""
Module defines sets up Adam optimization for a keras model
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics
    Args:
        network is the model to optimize
        alpha is the learning rate
        beta1 is the first Adam optimization parameter
        beta2 is the second Adam optimization parameter
    Returns:
        None
    """
    network.compile(
        optimizer=K.optimizers.Adam(alpha, beta1, beta2),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return None
