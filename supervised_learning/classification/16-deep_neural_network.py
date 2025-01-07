#!/usr/bin/env python3
"""deep_neural_network"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        Class constructor
        Args:
            nx: is the number of input features
            layers: is a list representing the number of nodes
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(map(lambda x: not isinstance(x, int) or x <= 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for m in range(1, self.L + 1):
            prev_nodes = nx if m == 1 else layers[m - 2]

            self.weights[f"W{m}"] = (np.random.randn(layers[m - 1],
                                     prev_nodes) *
                                     np.sqrt(2 / prev_nodes))
            self.weights[f"b{m}"] = np.zeros((layers[m - 1], 1))
