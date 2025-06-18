#!/usr/bin/env python3
"""
Transformer Decoder
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder
    """

    def __init__(
        self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1
    ):
        """
        Initializes the Decoder.
        Args:
            N (int): Number of blocks in the decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of heads in the multi-head attention.
            hidden (int): Dimensionality of the feed-forward network.
            target_vocab (int): Size of the target vocabulary.
            max_seq_len (int): Maximum sequence length for positional encoding.
            drop_rate (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N

        # Embedding layer for target vocabulary
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)

        # Positional encoding matrix
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List of DecoderBlocks
        self.blocks = (
            [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        )

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass of the decoder.
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, target_seq_len).
            encoder_output (tf.Tensor): Output from the encoder.
            training (bool): Whether the model is in training mode.
            look_ahead_mask (tf.Tensor): Mask for look-ahead attention.
            padding_mask (tf.Tensor): Mask for padding tokens.
        Returns:
            tf.Tensor: Output tensor after passing through the decoder.
        """
        seq_len = tf.shape(x)[1]

        # Embedding + scaling
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each decoder block
        for i in range(self.N):
            x = self.blocks[i](
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

        return x
