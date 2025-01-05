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

    def forward_prop(self, X):
        """
        Calculate the forward propagation
        Args:
            X:numpy.ndarray with shape (nx, m) that contains the input data:
                nx is the number of input features to the neuron
                m is the number of examples
        Returns:
            the private attribute __A
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y is a numpy.ndarray with shape (1, m)
            A is a numpy.ndarray with shape (1, m)
        Return
            cost
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost
