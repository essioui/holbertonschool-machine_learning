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

        for s in range(1, self.__L + 1):
            W = self.__weights[f"W{s}"]
            b = self.__weights[f"b{s}"]
            A_prev = self.__cache[f"A{s - 1}"]

            z1 = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-z1))

            self.__cache[f"A{s}"] = A
        return self.__cache[f"A{self.__L}"], self.__cache

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
        Evaluates the neural network’s predictions
        """
        A, _ = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost_value = self.cost(Y, A)

        return prediction, cost_value

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]

        L = self.__L
        weights = self.__weights
        dz = cache[f"A{L}"] - Y

        for s in range(L, 0, -1):

            dW = (1 / m) * np.dot(cache[f"A{s - 1}"], dz.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.dot(self.__weights[f"W{s}"].T, dz) * (cache[f"A{s - 1}"] * (1 - cache[f"A{s - 1}"]))

            self.__weights[f"W{s}"] -= (alpha * dW).T
            self.__weights[f"b{s}"] -= alpha * db

