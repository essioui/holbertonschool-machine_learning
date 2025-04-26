#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    Args:
        input_dims is a tuple of integers containing the dimensions
        filters is a list containing the number of filters for each
            convolutional layer in the encoder, respectivel
        latent_dims is a tuple of integers contain the dimensions latent space
    Returns:
        encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder mode
    should be compiled using adam optimization and binary cross-entropy loss
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)

    encoder_input = inputs

    for f in filters:
        encoder_input = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(encoder_input)

        encoder_input = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(encoder_input)

    # Decoder
    dec_inputs = keras.Input(shape=latent_dims)

    decoder_input = dec_inputs

    reversed_filters = filters[:-1][::-1]

    for f in reversed_filters:
        decoder_input = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(decoder_input)
        decoder_input = keras.layers.UpSampling2D(size=(2, 2))(decoder_input)

    # Special last two layers
    decoder_input = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(decoder_input)

    decoder_input = keras.layers.UpSampling2D(size=(2, 2))(decoder_input)

    outputs = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(decoder_input)

    # Models
    encoder = keras.Model(inputs=inputs, outputs=encoder_input)

    decoder = keras.Model(inputs=dec_inputs, outputs=outputs)

    auto_input = keras.Input(shape=input_dims)

    encoded = encoder(auto_input)

    decoded = decoder(encoded)

    auto = keras.Model(inputs=auto_input, outputs=decoded)

    # Compile models
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    encoder.compile(optimizer='adam', loss='binary_crossentropy')

    decoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
