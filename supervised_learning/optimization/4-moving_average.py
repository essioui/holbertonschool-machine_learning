#!/usr/bin/env python3
"""
Defines module Moving Average
"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
    Args:
        data is the list of data to calculate the moving average of
        beta is the weight used for the moving average
    Returns:
        list containing the moving averages of data
    """
    S_t = 0
    moving_average = []
    for i in range(len(data)):
        S_t = beta * S_t + (1 - beta) * data[i]
        bias_correction = 1 - beta ** (i + 1)
        correct_average = S_t / bias_correction
        moving_average.append(correct_average)
    return moving_average
