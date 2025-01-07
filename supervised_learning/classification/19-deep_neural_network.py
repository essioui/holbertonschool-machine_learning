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
        private instance attributes
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(map(lambda x: not isinstance(x, int) or x <= 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for m in range(1, self.L + 1):
            prev_nodes = nx if m == 1 else layers[m - 2]

            self.weights[f"W{m}"] = (np.random.randn(layers[m - 1],
                                     prev_nodes) *
                                     np.sqrt(2 / prev_nodes))
            self.weights[f"b{m}"] = np.zeros((layers[m - 1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache['A0'] = X

        for m in range(1, self.__L + 1):
            W = self.__weights[f"W{m}"]
            b = self.__weights[f"b{m}"]
            A_prev = self.__cache[f"A{m - 1}"]

            z1 = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-z1))

            self.__cache[f"A{m}"] = A
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - A) * np.log(1.0000001 - A))
        return cost
