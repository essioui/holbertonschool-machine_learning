#!/usr/bin/env python3
"""
Module defines Save and Load Model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model:
    Args:
        network is the model to save
        filename is the path of the file that the model should be saved to
    Returns:
        None
    """
    network.save(filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    loads an entire model
    Args:
        filename is the path of the file
    Returns:
        the loaded model
    """
    model = K.models.load_model(filename)
    print(f"Model loaded to {filename}")
    return model
