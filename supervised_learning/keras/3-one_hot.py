#!/usr/bin/env python3
"""
Module defines converts a label vector into a one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
    Args:
        labels [vector]:
            contains labels to convert into one-hot matrix
        classes:
            classes for one-hot matrix
    last dimension of the one-hot matrix must be the number of classes
    returns:
        one-hot matrix
    """
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
