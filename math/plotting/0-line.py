#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    line() generates a line plot with the x-axis ranging from 0 to 10 and 
    the y-axis being the cube of the x values.

    The function:
    - Initializes x values ranging from 0 to 10 using np.arange(0, 11)
    - Calculates the y values as the cube of the x values
    - Creates a figure with specified dimensions
    - Plots the x and y values as a red solid line
    - Sets the x-axis limits from 0 to 10 using plt.xlim(0, 10)
    - Displays the plot using plt.show()

    Returns:
    None
    """
    y = np.arange(0, 11) ** 3
    # Create a figure with specific dimensions
    plt.figure(figsize=(6.4, 4.8))

    # Plot the x and y values as a red solid line
    plt.plot(np.arange(0, 11), y, 'r')
    # Set the x-axis limits to range from 0 to 10
    plt.xlim(0, 10)
    # Display the plot
    plt.show()
