#!/usr/bin/env python3
"""
Clustering module
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset
    Args:
        X is a numpy.ndarray of shape (n, d)
        k is the number of cluster
    Returns:
        pi, m, S, clss, bic
            pi is a numpy.ndarray of shape (k,)
            m is a numpy.ndarray of shape (k, d)
            S is a numpy.ndarray of shape (k, d, d)
            clss is a numpy.ndarray of shape (n,)
            bic is a numpy.ndarray of shape (kmax - kmin + 1)
    """

    Gaussian = sklearn.mixture.GaussianMixture(n_components=k)
    params = Gaussian.fit(X)
    clss = Gaussian.predict(X)
    pi = params.weights_
    m = params.means_
    S = params.covariances_
    bic = Gaussian.bic(X)

    return pi, m, S, clss, bic
