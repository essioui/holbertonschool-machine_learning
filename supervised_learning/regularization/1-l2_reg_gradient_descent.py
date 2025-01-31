#!/usr/bin/env python3
"""
Module defines Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using
    gradient descent with L2 regularization
    Args:
        - Y is a one-hot numpy.ndarray of shape (classes, m)
            + classes is the number of classes
            + m is the number of data points
        weights is a dictionary of the weights and biases of nn
        cache is a dictionary of the outputs of each layer of nn
        alpha is the learning rate
        lambtha is the L2 regularization parameter
        L is the number of layers of the network
    """
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer - 1}'] if layer > 1 else cache['A0']
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        d_tanh = 1 - np.square(A_prev)
        dZ = np.dot(weights[f'W{layer}'].T, dZ) * d_tanh
        reg_l2 = (1 - lambtha * alpha / m)
        weights[f'W{layer}'] = reg_l2 * weights[f'W{layer}'] - alpha * dW
        weights[f'b{layer}'] -= alpha * db
