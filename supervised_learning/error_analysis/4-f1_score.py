#!/usr/bin/env python3
"""
Defines modules F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix
    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
    Returns:
        numpy.ndarray of shape (classes,)
    """
    precis = precision(confusion)
    sensitivi = sensitivity(confusion)
    F1 = 2 * precis * sensitivi / (precis + sensitivi)
    return F1
