#!/usr/bin/env python3
"""
Bayesian Optimization module
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor
        Args:
            f is the black-box function to be optimized
            X_init is a numpy.ndarray of shape (t, 1)
            Y_init is a numpy.ndarray of shape (t, 1)
            t is the number of initial samples
            bounds is a tuple of (min, max)
            ac_samples is the number of samples
            l is the length parameter for the kernel
            sigma_f is the standard deviation
            xsi is the exploration-exploitation factor for acquisition
            minimize is a bool determining whether optimization
        You may use GP = __import__('2-gp').GaussianProcess
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.bounds = bounds
        self.xsi = xsi
        self.minimize = minimize

        X_min, X_max = bounds
        self.X_s = np.linspace(X_min, X_max, ac_samples).reshape(-1, 1)
