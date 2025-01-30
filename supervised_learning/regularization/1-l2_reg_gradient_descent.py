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
    act_last_layer = cache[f'A{L}']
    dz_softmax = act_last_layer - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer - 1}'] if layer > 1 else cache['A0']
        W = weights[f'W{layer}']

        dW = (np.dot(dz_softmax, A_prev.T) / m) + ((lambtha / m) * W)

        weights[f'W{layer}'] -= alpha * dW

        if layer > 1:
            dA_prev = np.dot(W.T, dz_softmax)
            dz_softmax = dA_prev * (1 - A_prev ** 2)
