#!/usr/bin/env python3
"""
Module represents a poisson distribution
"""


class Poisson:
    """
    A class representing a Poisson distribution.
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Constructor to initialize the Poisson distribution.
        Args:
            data: A list of data points (optional).
            lambtha: Expected number of occurrences (positive float).
        raise:
            ValueError: If `lambtha` is not a positive value.
            TypeError: If `data` is not a list.
            ValueError: If `data` does not contain multiple values
        """
        e = 2.7182818285

        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")

        self.lambtha = float(lambtha)

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Calculates the probability mass function (PMF) for a given k value.
        The PMF of a Poisson distribution is calculated using the formula:
        P(X = k) = (lambda^k * e^(-lambda)) / k!
        Arguments:
        k (int):
            The number of occurrences for which the probability is calculated.
        Returns:
            float: The probability of having exactly `k` occurrences.
        Raises:
        ValueError: If k is not a non-negative integer.
        """
        k = int(k)

        if k < 0:
            return 0

        lambtha = self.lambtha
        exp_neg_lambda = Poisson.e ** (-lambtha)

        result = (lambtha ** k) * exp_neg_lambda / factorial(k)
        
        return result
