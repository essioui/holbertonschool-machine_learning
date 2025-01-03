#!/usr/bin/env python3
"""
Privatize Neuron
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
        Private instance attributes:
            __W
            __b
            __A
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
