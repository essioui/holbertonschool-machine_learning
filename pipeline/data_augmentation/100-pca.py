#!/usr/bin/env python3
"""
PCA Color Augmentation
"""
import tensorflow as tf

def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image as in AlexNet paper.

    Args:
        image (tf.Tensor): 3D tensor of shape (H, W, 3)
        alphas (tuple or np.ndarray): length-3 vector for channel changes

    Returns:
        tf.Tensor: Augmented image of same shape and dtype uint8
    """
    image = tf.cast(image, tf.float32)  # keep values in 0-255
    orig_shape = tf.shape(image)
    
    # Flatten image to (num_pixels, 3)
    flat_image = tf.reshape(image, [-1, 3])
    
    # Compute mean and center
    mean = tf.reduce_mean(flat_image, axis=0)
    centered = flat_image - mean

    # Covariance matrix
    cov = tf.matmul(tf.transpose(centered), centered) / tf.cast(tf.shape(flat_image)[0], tf.float32)
    
    # Eigen decomposition
    eigvals, eigvecs = tf.linalg.eigh(cov)
    
    # Delta adjustment for each channel
    alphas = tf.constant(alphas, dtype=tf.float32)
    delta = tf.matmul(eigvecs, tf.reshape(eigvals * alphas, [3,1]))
    delta = tf.reshape(delta, [1,3])  # add once per channel

    # Apply augmentation
    augmented = flat_image + delta
    augmented = tf.clip_by_value(augmented, 0, 255)
    
    # Reshape back to original
    return tf.cast(tf.reshape(augmented, orig_shape), tf.uint8)
