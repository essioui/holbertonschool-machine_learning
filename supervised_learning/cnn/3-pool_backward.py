#!/usr/bin/env python3
"""
Module Pooling Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
    Args:
        dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
        kernel_shape is a tuple of (kh, kw)
            kh is the kernel height
            kw is the kernel width
        stride is a tuple of (sh, sw)
            sh is the stride for the height
            sw is the stride for the width
        mode is a string containing either max or avg
    Returns:
        the partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    h_prev, w_prev, c = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        A_slice = A_prev[i, vert_start:vert_end,
                                         horiz_start:horiz_end, c]

                        mask = (A_slice == np.max(A_slice))

                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                c] += mask * dA[i, h, w, c]

                    elif mode == 'avg':
                        avg_grad = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                c] += np.full((kh, kw), avg_grad)

    return dA_prev
