#!/usr/bin/env python3
"""
Module Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        kernel_shape: tuple (kh, kw) for kernel size
        stride: tuple (sh, sw) for stride values
        mode: 'max' or 'avg' for pooling type
    Returns:
        Output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # حساب أبعاد المصفوفة الناتجة
    h = (h_prev - kh) // sh + 1
    w = (w_prev - kw) // sw + 1

    # تهيئة مصفوفة الخرج
    A = np.zeros((m, h, w, c_prev))

    for i in range(h):  # لكل موضع رأسي
        for j in range(w):  # لكل موضع أفقي
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A
