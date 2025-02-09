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
        new_heigt = (((h - 1) * sh + kh - h) // 2) + 1
        new_width = (((w - 1) * sw + kw - w) // 2) + 1
    if padding == 'valid':
        new_heigt, new_width = 0, 0
    if type(padding) is tuple:
        new_heigt, new_width = padding
    imgh, imgw = (h-kh+2*new_heigt)//sh + 1, (w-kw+2*new_width)//sw + 1
    output = np.zeros((m, imgh, imgw))
    new = np.pad(images, ((0, 0), (new_heigt, new_heigt),
                          (new_width, new_width)), 'constant')
    for i in range(imgh):
        for j in range(imgw):
            output[:, i, j] = np.tensordot(new[:,
                                           i*sh:i*sh+kh,
                                           j*sw:j*sw+kw],
                                           kernel)
    return output
