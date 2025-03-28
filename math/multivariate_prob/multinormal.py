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

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_center = data - self.mean

        self.cov = (data_center @ data_center.T) / (n - 1)
