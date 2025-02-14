#!/usr/bin/env python3
"""
Module Pooling Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    Args:
        dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
            kh is the filter height
            kw is the filter width
        b is a numpy.ndarray of shape (1, 1, 1, c_new)
        padding is a string that is either same or valid
        stride is a tuple of (sh, sw)
            sh is the stride for the height
            sw is the stride for the width
    Returns:
        the partial derivatives dA_prev, dW, db
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = max((h_prev - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_prev - 1) * sw + kw - w_prev, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        A_prev_pad = np.pad(A_prev, ((0,), (pad_top,),
                                     (pad_left,), (0,)), mode='constant')
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0
        A_prev_pad = A_prev

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    dW = np.zeros(W.shape)

    dA_prev_pad = np.pad(np.zeros(A_prev.shape), ((0,), (pad_top,),
                                                  (pad_bottom,), (pad_left,),
                                                  (pad_right,), (0,)),
                         mode='constant')
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    dW[:, :, :, c] += A_prev_pad[i, vert_start:vert_end,
                                                 horiz_start:horiz_end,
                                                 :] * dZ[i, h, w, c]

                    dA_prev_pad[i, vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]

    dA_prev = dA_prev_pad[:, pad_top:pad_top+h_prev,
                          pad_left:pad_left+w_prev, :]

    return dA_prev, dW, db
