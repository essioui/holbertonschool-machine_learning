#!/usr/bin/env python3
"""
Module define Pooling
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    Args:
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layerc_prev
            c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b is a numpy.ndarray of shape (1, 1, 1, c_new)
        activation is an activation function applied to the convolution
        padding is a string that is either same or valid
        stride is a tuple of (sh, sw)
            sh is the stride for the height
            sw is the stride for the width
        Returns:
            the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1

    A_prev_padding = np.pad(A_prev, ((0, 0), (ph, ph),
                                     (pw, pw), (0, 0)), 'constant')

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw
                A = A_prev_padding[:, h_start:h_end, w_start:w_end, :]
                Z[:, i, j, k] = np.sum(A * W[..., k],
                                       axis=(1, 2, 3)) + b[..., k]
    return activation(Z)
