#!/usr/bin/env python3
"""
 single neuron performing binary
"""
import numpy as np


class Neuron:
    """
     Defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Constructor build single neuron
        Args:
            nx: the number of input features to the neuron
        Raises:
            TypeError("nx must be an integer")
            ValueError("nx must be a positive")
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
