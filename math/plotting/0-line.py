#!/usr/bin/env python3
"""plot y as a line graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    line graph
    plt.figure: dimension of figure (width, height)
    plt.plot(x, y, 'r) plots the x and y values with a red solid line:
        x = np.arange(0, 11)
        y = np.arange(0,11) ** 3
        plt.xlim = the axis x start from 0 end 10
        r: for color red
    show(): function for see the figure
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r')
    plt.xlim(0, 10)
    plt.show()
