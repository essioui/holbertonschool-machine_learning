#!/usr/bin/env python3
"""
Bayesian optimization
"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        init method for bayesian optimization
        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
                    t: the number of initial samples
            bounds: tuple of (min, max) representing the bounds of the space
                    in which to look for the optimal point
            ac_samples: the number of samples that should be analyzed during
                        acquisition
            l: the length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
                     black-box function
            xsi: the exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be performed
                      for minimization (True) or maximization (False)
        """
        # black-box function
        self.f = f

        # Gaussian Process
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # X_s all acquisition sample
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)

        # exploration-explotation
        self.xsi = xsi

        # minimization versus maximization
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
            Uses the Expected Improvement acquisition function
        Returns:
            X_next, EI
                X_next is a numpy.ndarray of shape (1,)
                EI is a numpy.ndarray of shape (ac_samples,)
        You may use from scipy.stats import norm
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)

        if self.minimize:
            best = np.min(self.gp.Y)
            improv = best - mu.reshape(-1, 1) - self.xsi
        else:
            best = np.max(self.gp.Y)
            improv = mu.reshape(-1, 1) - best - self.xsi

        Z = np.zeros_like(improv)
        with np.errstate(divide='warn'):
            Z[sigma > 0] = improv[sigma > 0] / sigma[sigma > 0]

        EI = improv * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next.reshape(1,), EI.reshape(-1)

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        Args:
            iterations is the maximum number of iterations to perform
        Returns:
            X_opt, Y_opt
                X_opt is a numpy.ndarray of shape (1,)
                Y_opt is a numpy.ndarray of shape (1,)
        """
        X = []
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            X.append(X_next)

        return X_next, Y_next
