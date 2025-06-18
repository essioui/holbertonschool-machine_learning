#!/usr/bin/env python3
"""
Transformer Network
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer Network
    """
    def __init__(
        self, N, dm, h, hidden, input_vocab, target_vocab,
        max_seq_input, max_seq_target, drop_rate=0.1
    ):
        """
        Initializes the Transformer.
        Args:
            N (int): Number of blocks in the encoder and decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of heads in the multi-head attention.
            hidden (int): Dimensionality of the feed-forward network.
            input_vocab (int): Size of the input vocabulary.
            target_vocab (int): Size of the target vocabulary.
            max_seq_input (int): Maximum sequence length for input.
            max_seq_target (int): Maximum sequence length for target.
            drop_rate (float): Dropout rate.
        """
        super(Transformer, self).__init__()

        # Initialize the encoder and decoder
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )

        # Decoder takes the same parameters as the encoder
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )

        # Linear layer for final output
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self, inputs, target, training, encoder_mask,
        look_ahead_mask, decoder_mask
    ):
        """
        Forward pass of the Transformer.
        Args:
            inputs: Input tensor of shape (batch_size, input_seq_len).
            target: Target tensor of shape (batch_size, target_seq_len).
            training (bool): Whether the model is in training mode.
            encoder_mask (tf.Tensor): Mask for the encoder.
            look_ahead_mask (tf.Tensor): Look-ahead mask for the decoder.
            decoder_mask (tf.Tensor): Mask for the decoder.
        Returns:
            tf.Tensor:
            Output tensor of shape (batch_size, target_seq_len, target_vocab).
        """
        # Pass inputs through the encoder
        enc_output = self.encoder(
            inputs, training=training, padding_mask=encoder_mask
        )

        # Pass target through the decoder
        dec_output = self.decoder(
            target, enc_output, training=training,
            look_ahead_mask=look_ahead_mask, padding_mask=decoder_mask
        )

        # Final linear layer to produce output
        output = self.linear(dec_output)

        return output
