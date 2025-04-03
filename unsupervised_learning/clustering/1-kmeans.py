#!/usr/bin/env python3
"""Clustering module"""
import numpy as np


def get_closest(X, centroids):
    """Finds the closest points to centroids

    Arguments:
        X {np.ndarray} -- Containing the dataset
        centroids {np.ndarray} -- Containing the centroids in each dimension

    Returns:
        np.ndarray -- Containing the index of the nearest centroid
    """
    distances = np.sqrt(np.sum((X - centroids[:, np.newaxis])**2, axis=2))
    return np.argmin(distances, axis=0)


def kmeans(X, k, iterations=1000):
    """Performs the K-means algorithm

    Arguments:
        X {np.ndarray} -- Containing the data set
        k {int} -- Is the number of clusters

    Keyword Arguments:
        iterations {int} -- Is the number of iterations (default: {1000})

    Returns:
        np.ndarray, np.ndarray -- The newly moved centroids, and the newly
        assigned classes
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    n, d = X.shape
    centroids = np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, d)
    )

    for i in range(iterations):
        old_centroids = np.copy(centroids)
        closest = get_closest(X, centroids)

        for j in range(k):
            if (X[closest == j].size == 0):
                centroids[j] = np.random.uniform(
                    low=np.min(X, axis=0),
                    high=np.max(X, axis=0),
                    size=(1, d)
                )
            else:
                centroids[j] = np.mean(X[closest == j], axis=0)
        if np.all(old_centroids == centroids):
            break

    return centroids, get_closest(X, centroids)
