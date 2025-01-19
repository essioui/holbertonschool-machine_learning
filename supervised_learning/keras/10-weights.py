#!/usr/bin/env python3
"""
Module defines functions to save and load a model's weights using only built-in Python features
"""
import tensorflow.keras as K
import json


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights in the specified format.

    Args:
        network: The model whose weights should be saved.
        filename: The path of the file to save the weights.
        save_format: The format to save the weights in. Default is 'keras'.
                      'keras' saves weights in Keras default format (as nested lists).
                      'json' saves weights as a JSON file.
        
    Returns:
        None
    """
    weights = network.get_weights()  # Get the model's weights

    if save_format == 'keras':
        # Save as nested lists
        with open(filename, 'w') as f:
            for weight in weights:
                f.write(f"{weight.tolist()}\n")  # Convert weights to list and save
        print(f"Weights saved to {filename} in 'keras' format.")

    elif save_format == 'json':
        # Save as a JSON file
        with open(filename, 'w') as f:
            json.dump([weight.tolist() for weight in weights], f)  # Save weights as JSON
        print(f"Weights saved to {filename} in 'json' format.")
    
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
                      'json' loads from a JSON format.
        
    Returns:
        None
    """
    weights = []
    
    if load_format == 'keras':
        # Load weights from file
        with open(filename, 'r') as f:
            for line in f:
                weights.append(K.backend.constant(eval(line.strip())))  # Convert each line back to a tensor safely
        
    elif load_format == 'json':
        # Load weights from JSON file
        with open(filename, 'r') as f:
            loaded_weights = json.load(f)
            for weight in loaded_weights:
                weights.append(K.backend.constant(weight))  # Convert to tensor
        
    else:
        raise ValueError(f"Unsupported load format: {load_format}")

    # Load weights into the network
    network.set_weights(weights)
    print(f"Weights loaded from {filename} in '{load_format}' format.")
