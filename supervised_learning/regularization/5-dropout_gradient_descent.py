#!/usr/bin/env python3
"""
Module defines Gradient Descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout
    regularization using gradient descent
    Args:
        Y is a one-hot numpy.ndarray of shape (classes, m)
            classes is the number of classes
            m is the number of data points
        weights is a dictionary of the weights and biases
        cache is a dictionary of the outputs and dropout masks of each layer
        alpha is the learning rate
        keep_prob is the probability that a node will be kept
        L is the number of layers of the network
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"] if i > 1 else cache["A0"]
        W = weights[f"W{i}"]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA = np.dot(W.T, dZ)
            dA *= cache[f"D{i-1}"]
            dA /= keep_prob
            dZ = dA * (1 - cache[f"A{i-1}"] ** 2)

        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
