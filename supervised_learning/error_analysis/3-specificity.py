#!/usr/bin/env python3
"""
Defines modules Specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            classes is the number of classes
    Returns:
        numpy.ndarray of shape (classes,)
    """
    true_positives = np.diag(confusion)
    total = np.sum(confusion)
    rows_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)
    true_negative = total - (col_sums + rows_sums) + true_positives

    false_positives = col_sums - true_positives

    specifi = true_negative / (true_negative + false_positives)
    return specifi
