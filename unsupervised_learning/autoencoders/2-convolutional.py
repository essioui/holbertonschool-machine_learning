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
    encoder_input = keras.Input(shape=input_dims)

    X = encoder_input

    for f in filters:
        X = keras.layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding="same", activation="relu"
            )(X)

        X = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(X)

    latent = keras.layers.Conv2D(
        filters=latent_dims[-1], kernel_size=(3, 3),
        padding="same", activation="relu"
        )(X)

    encoder = keras.models.Model(inputs=encoder_input, outputs=latent)

    # Decoder
    decoder_input = keras.layers.Input(shape=latent_dims)

    X = decoder_input

    for f in reversed(filters[:-1]):
        X = keras.layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding="same", activation="relu"
        )(X)

        X = keras.layers.UpSampling2D(size=(2, 2))(X)

    X = keras.layers.Conv2D(
        filters=filters[0], kernel_size=(3, 3),
        padding="valid", activation="relu"
    )(X)

    X = keras.layers.Conv2D(
        filters=input_dims[-1], kernel_size=(3, 3),
        padding="same", activation="sigmoid"
    )(X)

    X = keras.layers.UpSampling2D(size=(2, 2))(X)

    decoder = keras.models.Model(inputs=decoder_input, outputs=X)

    # Autoencoder
    auto_input = encoder_input

    encoded = encoder(auto_input)

    decoded = decoder(encoded)

    auto = keras.models.Model(inputs=auto_input, outputs=decoded)

    # Model
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
