#!/usr/bin/env python3
"""
Module define hyperparameter
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix,
        the kernel should use the Radial Basis Function (RBF)
        Args:
            X1 is a numpy.ndarray of shape (m, 1)
            X2 is a numpy.ndarray of shape (n, 1
        Returns:
            the covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1) + np.sum(
                X2**2, 1) - 2 * np.dot(X1, X2.T)
        )
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in Gaussian process
        Args:
            X_s is a numpy.ndarray of shape (s, 1)
                s is the number of sample points
        Returns:
            mu, sigma
                mu is a numpy.ndarray of shape (s,)
                sigma is a numpy.ndarray of shape (s,)
        """
        K_s = self.kernel(self.X, X_s)

        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(self.K)

        mean = K_s.T @ K_inv @ self.Y
        mu = mean.reshape(-1)

        cov = K_ss - K_s.T @ K_inv @ K_s

        sigma = np.diag(cov)

        return mu, sigma
