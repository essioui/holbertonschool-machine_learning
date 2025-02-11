#!/usr/bin/env python3
"""
Module define Multiple Kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels
    Args:
        images is a numpy.ndarray with shape (m, h, w, c)
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernels is a numpy.ndarray with shape (kh, kw, c, nc)
            kh is the height of a kernel
            kw is the width of a kernel
            nc is the number of kernels
        padding is either a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

    images_padded = np.pad(images, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), mode='constant')

    output = np.zeros((m, new_h, new_w, nc))

    for i in range(new_h):
        for j in range(new_w):
            x_start, y_start = i * sh, j * sw
            x_end, y_end = x_start + kh, y_start + kw
            output[:, i, j, :] = np.tensordot(
                images_padded[:, x_start:x_end, y_start:y_end, :],
                kernels, axes=([1, 2, 3], [0, 1, 2]))

    return output
