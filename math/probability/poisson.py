#!/usr/bin/env python3
"""
Module represents a poisson distribution
"""


class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """
        Constructor to initialize the Poisson distribution.
        Args:
            data: A list of data points (optional).
            lambtha: Expected number of occurrences (positive float).
        """
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

        self.lambtha = float(lambtha)

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = sum(data) / len(data)
