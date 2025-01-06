#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer
"""

import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dW2 = np.dot(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train NeuralNetwork
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iteration_list = []

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            _, cost = self.evaluate(X, Y)

            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")

            if graph and i % step == 0:
                costs.append(cost)
                iteration_list.append(i)
            self.gradient_descent(X, Y, A1, A2, alpha)

        if graph:
            plt.plot(iteration_list, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
