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
        stride is a tuple of (sh,sw)
            sh is the stride for the height
            sw is the stride for the width
    Returns:
        the partial derivatives dA_prev, dW, db
    """
    m, imgh, imgw, c = dZ.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    imghp, imgwp = 0, 0
    if padding == 'same':
        imghp = (((imgh * sh) - sh + kh - imgh) // 2) + 1
        imgwp = (((imgw * sw) - sw + kw - imgw) // 2) + 1
    if isinstance(padding, tuple):
        imghp, imgwp = padding

    new = np.pad(A_prev, ((0, 0), (imghp, imghp),
                          (imgwp, imgwp), (0, 0)),
                 'constant', constant_values=0)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    newDZ = np.zeros(new.shape)
    dW = np.zeros_like(W)
    for n in range(m):
        for i in range(imgh):
            for j in range(imgw):
                for k in range(knc):
                    newDZ[n,
                          i*sh:i*sh+kh,
                          j*sw:j*sw+kw, :] += np.multiply(dZ[n, i, j, k],
                                                          W[..., k])
                    dW[..., k] += np.multiply(dZ[n, i, j, k],
                                              new[n,
                                                  i*sh:i*sh+kh,
                                                  j*sw:j*sw+kw, :])
    if padding == 'same':
        newDZ = newDZ[:, imghp:-imghp, imgwp:-imgwp, :]
    return newDZ, dW, db
