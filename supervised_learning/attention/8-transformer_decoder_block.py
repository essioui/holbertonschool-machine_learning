#!/usr/bin/env python3
"""
Transformer Decoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer Decoder Block
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the decoder block
        Args:
            dm: Dimensionality of the model
            h: Number of attention heads
            hidden: Dimensionality of the feed-forward network
            drop_rate: Dropout rate
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass of the decoder block
        Args:
            x: Input tensor of shape (batch_size, target_seq_len, dm)
            encoder_output: Output from the encoder of shape
                (batch_size, input_seq_len, dm)
            training: Boolean indicating whether the model is in training mode
            look_ahead_mask: Mask for look-ahead attention
            padding_mask: Mask for padding tokens
        Returns:
            Output tensor of shape (batch_size, target_seq_len, dm)
        """
        # Apply padding mask to encoder output
        attn_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(x + attn_output1)

        # Apply padding mask to encoder output
        attn_output2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(out1 + attn_output2)

        # Feed-forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layernorm3(out2 + ffn_output)

        return output
