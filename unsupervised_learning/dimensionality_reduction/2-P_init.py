#!/usr/bin/env python3
"""
Module define  Initialize t-SNE
"""
import numpy as np


def P_init(X: np.ndarray, perplexity: float):
    """
    Initializes all variables required to calculate the P affinities in t-SNE
    Args:
        X is a numpy.ndarray of shape (n, d)
            n is the number of data points
            d is the number of dimensions in each poin
        perplexity is the perplexity that all Gaussian distributions
    Returns:
        (D, P, betas, H
            D: a numpy.ndarray of shape (n, n)
            The diagonal of D should be 0s
            P: a numpy.ndarray of shape (n, n) initialized to all 0
            betas: a numpy.ndarray of shape (n, 1) initialized to all 1
    """
    n, d = X.shape

    sum_X = np.sum(np.square(X), axis=1, keepdims=True)

    D = sum_X + sum_X.T - 2 * np.dot(X, X.T)

    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
