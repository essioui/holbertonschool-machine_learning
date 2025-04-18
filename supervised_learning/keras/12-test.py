#!/usr/bin/env python3
"""
Module defines functions  Test
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    tests a neural network
    Args:
        network is the network model to test
        data is the input data to test the model with
        labels are the correct one-hot labels of data
        verbose is a boolean that determines if output should be
            printed during the testing process
    Returns:
        the loss and accuracy of the model with testing data, respectively
    """
    loss, accuracy = network.evaluate(x=data,
                                      y=labels,
                                      verbose=verbose)
    return [loss, accuracy]
