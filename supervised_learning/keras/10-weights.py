#!/usr/bin/env python3
"""
Module defines functions to save and load a model's weights using only built-in Python features
"""
import tensorflow.keras as K


def save_weights(network, filename):
    """
    Saves a model's weights using `get_weights`.
    Args:
        network: the model whose weights should be saved
        filename: the path of the file to save the weights
    Returns:
        None
    """
    weights = network.get_weights()  # Get the model's weights
    # Save the weights as nested lists
    with open(filename, 'w') as f:
        for weight in weights:
            # Save each weight's data and shape
            f.write(f"{weight.tolist()}\n")  # Convert weights to list and save
    print(f"Weights saved to {filename}.")


def load_weights(network, filename):
    """
    Loads a model's weights using `set_weights`.
    Args:
        network: the model to which the weights should be loaded
        filename: the path of the file to load the weights from
    Returns:
        None
    """
    weights = []
    # Load the weights from the file
    with open(filename, 'r') as f:
        for line in f:
            weights.append(K.backend.constant(eval(line.strip())))  # Convert each line back to a tensor safely using constant
    # Load weights into the network
    network.set_weights(weights)
    print(f"Weights loaded from {filename}.")
