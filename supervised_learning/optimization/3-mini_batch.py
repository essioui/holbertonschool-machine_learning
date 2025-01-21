#!/usr/bin/env python3
"""
Define module Mini-Batch
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural
    network using mini-batch gradient descent
    Args:
        X is a numpy.ndarray of shape (m, nx) representing input data
            m is the number of data points
            nx is the number of features in X
        Y is a numpy.ndarray of shape (m, ny) representing the labels
            m is the same number of data points as in X
            ny is the number of classes for classification tasks
        batch_size is the number of data points in a batch
    Returns:
        list of mini-batches containing tuples (X_batch, Y_batch)
    """
    X_shuffle, Y_shuffle = shuffle_data(X, Y)
    m = X.shape[0]

    for start_indx in range(0, m, batch_size):
        end_indx = min(start_indx + batch_size, m)
        X_batch = X_shuffle[start_indx:end_indx]
        Y_batch = Y_shuffle[start_indx:end_indx]
        yield X_batch, Y_batch
