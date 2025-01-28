#!/usr/bin/env python3
"""
Defines modules Create Confusion
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            - classes is the number of classes
    Returns:
        numpy.ndarray (classes,) containing the sensitivity of each class
    """
    rows_total = np.sum(confusion, axis=1)
    true_positives = np.diag(confusion)
    sensitivity_values = true_positives / rows_total
    return sensitivity_values
