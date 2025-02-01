#!/usr/bin/env python3
"""
Module defines Forward Propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
    Args:
        X is a numpy.ndarray of shape (nx, m) containing the input data
            nx is the number of input features
            m is the number of data points
        weights is a dictionary of the weights and biases
        L the number of layers in the network
        keep_prob is the probability that a node will be kept
    Returns:
        dictionary containing the outputs of each layer
    """
    cache = {"A0": X}

    for i in range(1, L):
        W = weights[f"W{i}"]
        b = weights[f"b{i}"]
        A_prev = cache[f"A{i-1}"]

        Z = np.dot(W, A_prev) + b

        A = np.tanh(Z)

        D = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(int)
        A *= D
        A /= keep_prob

        cache[f"A{i}"] = A
        cache[f"D{i}"] = D

    W = weights[f"W{L}"]
    b = weights[f"b{L}"]
    A_prev = cache[f"A{L-1}"]

    ZL = np.dot(W, A_prev) + b

    exp_ZL = np.exp(ZL - np.max(ZL, axis=0, keepdims=True))
    A_L = exp_ZL / np.sum(exp_ZL, axis=0, keepdims=True)

    cache[f"A{L}"] = A_L

    return cache
