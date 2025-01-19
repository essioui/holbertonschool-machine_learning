#!/usr/bin/env python3
"""
Module defines functions  Save and Load Configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format
    Args:
        network is the model whose configuration should be saved
        filename is the path of the file that the configuration
    Returns:
        None
    """
    json = network.to_json()
    with open(filename, 'w+') as f:
        f.write(json)
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration
    Args:
        filename is the path of the file containing
        the model’s configuration in JSON format
    Returns:
        the loaded model
    """
    with open(filename, 'r') as f:
        json_string = f.read()
    model = K.models.model_from_json(json_string)
    return model
