#!/usr/bin/env python3
"""
Module define Pooling
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    Args:
        images is a numpy.ndarray with shape (m, h, w, c)
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel_shape is a tuple of (kh, kw)
            kh is the height of the kernel
            kw is the width of the kernel
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling
    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1

    output = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            x_start, y_start = i * sh, j * sw
            x_end, y_end = x_start + kh, y_start + kw
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, x_start:x_end, y_start:y_end, :], axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    images[:, x_start:x_end, y_start:y_end, :], axis=(1, 2))

    return output
