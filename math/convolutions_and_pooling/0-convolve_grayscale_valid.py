#!/usr/bin/env python3
"""
Module define Valid Convolution
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images
    Args:
        images is a numpy.ndarray with shape (m, h, w)
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw)
            kh is the height of the kernel
            kw is the width of the kernel
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    new_height = h - kh + 1
    new_width = w - kw + 1

    convolution = np.zeros((m, new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            convolution[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                          axis=(1, 2))
    return convolution
