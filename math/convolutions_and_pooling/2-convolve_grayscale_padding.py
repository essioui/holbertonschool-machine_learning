#!/usr/bin/env python3
"""
Module define Convolution with Padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
        padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padding_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                            mode='constant', constant_values=0)

    new_height = h - kh + 2 * ph + 1
    new_width = w - kw + 2 * pw + 1

    convolution = np.zeros((m, new_height, new_width))

    for i in range(h):
        for j in range(w):
            convolution[:, i, j] = np.sum(padding_images[:, i:i+kh,
                                                         j:j+kw] * kernel,
                                          axis=(1, 2))
    return convolution
