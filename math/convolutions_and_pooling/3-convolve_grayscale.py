#!/usr/bin/env python3
"""
Module define Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images
    Args:
        images is a numpy.ndarray with shape (m, h, w)
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw)
            kh is the height of the kernel
            kw is the width of the kernel
        padding is either a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw - kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    new_height = (h - kh + 2 * ph) // sh + 1
    new_width = (w - kw + 2 * pw) // sw + 1

    images_padding = np.pad(images, ((0, 0), (ph, ph),
                                     (pw, pw)), mode='constant')

    convolution = np.zeros((m, new_height, new_width))

    for i in range(new_height):
        for j in range(new_width):
            region = images_padding[:, i * sh:i * sh + kh, j * sw:j * sw + kw]

            convolution[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return convolution
