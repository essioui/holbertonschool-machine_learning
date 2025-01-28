#!/usr/bin/env python3
"""
Defines modules Precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            - classes is the number of classes
    Returns:
        numpy.ndarray of shape (classes,)
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precisions = true_positives / (false_positives + true_positives)
    return precisions
