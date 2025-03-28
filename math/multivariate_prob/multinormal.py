#!/usr/bin/env python3
"""
Module define Mean and Initialize
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        Constructor for the MultiNormal class
        Args:
            data is a numpy.ndarray of shape (d, n):
                n is the number of data points
                d is the number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        self.d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_center = data - self.mean

        self.cov = (data_center @ data_center.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        Args:
            x is a numpy.ndarray of shape (d, 1):
                d is the number of dimensions of the Multinomial
        Returns:
            the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.d, 1):
            raise ValueError("x must have the shape ({d}, 1)")

        determinant_cov = np.linalg.det(self.cov)

        inv_cov = np.linalg.inv(self.cov)

        diff = x - self.mean

        mahalanobis_distance = diff.T @ inv_cov @ diff

        factor = 1 / (np.sqrt((2 * np.pi) ** self.d * determinant_cov))

        pdf_value = factor * np.exp(-0.5 * mahalanobis_distance)

        return pdf_value.item()
