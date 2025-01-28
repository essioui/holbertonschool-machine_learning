#!/usr/bin/env python3
"""
Defines modules Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    Args:
        labels is a one-hot numpy.ndarray of shape (m, classes)
            m is the number of data points
            classes is the number of classes
        logits is a one-hot numpy.ndarray of shape (m, classes)
    Returns:
        confusion numpy.ndarray of shape (classes, classes)
    """
    m = labels.shape[0]
    classes = labels.shape[1]

    confuson_matrix = np.zeros((classes, classes), dtype=float)

    for i in range(m):
        true_label = np.argmax(labels[i])
        pred_label = np.argmax(logits[i])
        confuson_matrix[true_label, pred_label] += 1
    return confuson_matrix
