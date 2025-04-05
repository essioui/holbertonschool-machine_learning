#!/usr/bin/env python3
"""
Clustering module
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset
    Args:
        X is a numpy.ndarray of shape (n, d)
        k is the number of clusters
    Returns:
        C, clss
            C is a numpy.ndarray of shape (k, d)
            clss is a numpy.ndarray of shape (n,)
    """
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)

    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss
