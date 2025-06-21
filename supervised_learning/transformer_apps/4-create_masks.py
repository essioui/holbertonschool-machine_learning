#!/usr/bin/env python3
"""
Create Masks for Transformer Models
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Create a padding mask for the input sequence.
    Args:
        seq (tf.Tensor): Input sequence tensor.
    Returns:
        tf.Tensor: Padding mask tensor.
    """
    # Create a mask where padding tokens (0) are set to 1
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for the decoder.
    Args:
        size (int): Size of the sequence.
    Returns:
        tf.Tensor: Look-ahead mask tensor.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Create masks for the inputs and target sequences.
    Args:
        inputs (tf.Tensor): Input sequence tensor.
        target (tf.Tensor): Target sequence tensor.
    Returns:
        encoder_mask, combined_mask, decoder_mask
    """
    # Padding mask for encoder
    encoder_mask = create_padding_mask(inputs)

    # Look-ahead mask + padding for target
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    decoder_mask = create_padding_mask(inputs)

    return encoder_mask, combined_mask, decoder_mask
