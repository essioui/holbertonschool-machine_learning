#!/usr/bin/env python3
"""
Sparse Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    """
    # Encoder
    input_layer = keras.Input(shape=(input_dims,))
    X = input_layer
    for nodes in hidden_layers:
        X = keras.layers.Dense(nodes, activation="relu")(X)

    encoded = keras.layers.Dense(
        latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(X)

    encoder = keras.Model(inputs=input_layer, outputs=encoded)

    # Decoder
    encoded_input = keras.Input(shape=(latent_dims,))
    X = encoded_input
    for nodes in reversed(hidden_layers):
        X = keras.layers.Dense(nodes, activation="relu")(X)

    decoded = keras.layers.Dense(input_dims, activation="sigmoid")(X)

    decoder = keras.Model(inputs=encoded_input, outputs=decoded)

    # Autoencoder
    auto_input = input_layer
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(inputs=auto_input, outputs=decoded_output)

    # Compile
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
