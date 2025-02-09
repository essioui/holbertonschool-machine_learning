#!/usr/bin/env python3
"""
Module define Convolution with Channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels
    Args:
        images is a numpy.ndarray with shape (m, h, w, c)
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel is a numpy.ndarray with shape (kh, kw, c)
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
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if c != kc:
        raise ValueError("number channels in image must match in kernel")

    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding
    else:
        raise ValueError("Invalid padding type")

    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, new_h, new_w))

    padded_images = np.pad(images, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), mode='constant')

    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(
                padded_images[:, i * sh:i * sh + kh,
                              j * sw:j * sw + kw, :] * kernel,
                axis=(1, 2, 3)
            )

    return output
