#!/usr/bin/env python3
"""
Module defines functions to save and load a model's weights using only built-in Python features.
This version does not use `json` or `numpy`.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights in the specified format.

    Args:
        network: The model whose weights should be saved.
        filename: The path of the file to save the weights.
        save_format: The format to save the weights in. Default is 'keras'.
                      'keras' saves weights in Keras default format (as nested lists).
        
    Returns:
        None
    """
    weights = network.get_weights()  # Get the model's weights

    if save_format == 'keras':
        # Save as nested lists (no external libraries, using Python list representation)
        with open(filename, 'w') as f:
            for weight in weights:
                # Save each weight matrix as a string representation of a list
                f.write(f"{weight.tolist()}\n")  
        print(f"Weights saved to {filename} in 'keras' format.")
    
    else:
        raise ValueError(f"Unsupported save format: {save_format}")


def load_weights(network, filename, load_format='keras'):
    """
    Loads a model's weights using `set_weights`.
    
    Args:
        network: The model to which the weights should be loaded.
        filename: The path of the file to load the weights from.
        load_format: The format the weights are saved in. Default is 'keras'.
                      'keras' loads from the format saved as nested lists.
        
    Returns:
        None
    """
    weights = []
    
    if load_format == 'keras':
        # Load weights from file (assuming they were saved in a nested list format)
        with open(filename, 'r') as f:
            for line in f:
                # Convert the string representation of a list back into an actual list of floats
                weights.append(K.backend.constant(eval(line.strip())))  # Safely evaluate the list
        
    else:
        raise ValueError(f"Unsupported load format: {load_format}")

    # Load weights into the network
    network.set_weights(weights)
    print(f"Weights loaded from {filename} in '{load_format}' format.")
