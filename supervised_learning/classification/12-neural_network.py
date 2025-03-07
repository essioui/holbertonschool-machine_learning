#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer
"""

import numpy as np


class NeuralNetwork:
    """
    neural network with one hidden layer performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        Neural network with one hidden layer
            nx is the number of input features
            nodes is the number of nodes found in the hidden layer
        W1: is array from nodes to nx
        W2: is array between 1 and nodes
        W2 come after W1
        Private instance attributes:
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        NeuralNetwork Forward Propagation
        """
        z1 = np.dot(self.__W1, X) + self.__b1

        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = np.dot(self.__W2, self.__A1) + self.__b2

        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        beta = 1.0000001 - A
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(beta))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        Args:
            X is a numpy.ndarray with shape (nx, m):
                nx is the number of input features to the neuron
                m is the number of examples
            Y is a numpy.ndarray with shape (1, m)
        Retrn:
             prediction should be a numpy.ndarray with shape (1, m)
             label values should be 1 if the output >= 0.5 and 0 otherwise
        """
        _, A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost_value = self.cost(Y, A)

        return prediction, cost_value
