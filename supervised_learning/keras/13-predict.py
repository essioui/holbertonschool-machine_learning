#!/usr/bin/env python3
"""
Module defines functions Predict
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network
    Args:
        network is the network model to make the prediction with
        data is the input data to make the prediction with
        verbose is a boolean that determines if output should be
            printed during the prediction process
    Returns:
        the prediction for the data
    """
    prediction = network.predict(x=data,
                                 verbose=verbose)
    return prediction
