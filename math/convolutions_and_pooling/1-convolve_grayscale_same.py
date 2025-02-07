#!/usr/bin/env python3
"""
Module define Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
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

    padding_height = kh // 2
    padding_width = kw // 2

    padding_images = np.pad(images, ((0, 0), (padding_height, padding_height),
                                     (padding_width, padding_width)),
                            mode='constant')
    convolution = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            convolution[:, i, j] = np.sum(
                padding_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    return convolution
