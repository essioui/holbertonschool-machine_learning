#!/usr/bin/env python3
"""
Module defines functions to save and load a model's weights
"""
import tensorflow.keras as K
import numpy as np


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model’s weights to a file or model.
    Args:
        network: the model whose weights should be saved
        filename: the path of the file that the weights should be saved to
        save_format: format of the saved file
    """
    if save_format == 'keras':
        network.save(filename)
    elif save_format == 'npy':
        weights = network.get_weights()
        np.save(filename, weights)
    else:
        raise ValueError("Unsupported save format. Choose 'keras' or 'npy'.")


def load_weights(network, filename, save_format='keras'):
    """
    Loads a model’s weights from a file
    Args:
        network: the model to which the weights should be loaded
        filename: the path of the file that the weights should be loaded from
        save_format:format in which the weights were saved, default is 'keras'
    Returns:
        None
    """
    if save_format == 'keras':
        network.load_weights(filename)
    elif save_format == 'npy':
        weights = np.load(filename, allow_pickle=True)
        network.set_weights(weights)
    else:
        raise ValueError("Unsupported load format. Choose 'keras' or 'npy'.")
    return None
