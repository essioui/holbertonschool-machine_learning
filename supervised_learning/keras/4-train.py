#!/usr/bin/env python3
"""
Module defines trains a model using mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent
    Args:
        network is the model to train
        data is a numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.of shape (m, classes) containing labels data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes data for mini-batch gradient descent
        verbose: boolean that determines if output printed during training
        shuffle: boolean that determine to shuffle the batches every epoch
    Returns:
        the History object generated after training the model
    """
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle
                          )
    return history
