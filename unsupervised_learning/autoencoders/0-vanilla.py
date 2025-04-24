#!/usr/bin/env python3
"""
Vanilla" Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    Args:
        input_dims is an integer containing the dimensions of the model input
        hidden_layers is a list containing the number of nodes
        latent_dims is an integer containing the dimensions 
    Returns:
        encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model
    * The autoencoder model should be compiled using adam optimization
      and binary cross-entropy loss
    * All layers should use a relu activation except for the last layer in
      the decoder, which should use sigmoid
    """
    input_layer = keras.Input(shape=(input_dims, ))

    X = input_layer

    for nodes in hidden_layers:
        X = keras.layers.Dense(nodes, activation="relu")(X)

    encoded = keras.layers.Dense(latent_dims, activation="relu")(X)
    
    encoder = keras.Model(inputs=input_layer, outputs=encoded)

    encoded_input = keras.Input(shape=(latent_dims, ))

    X = encoded_input

    for nodes in reversed(hidden_layers):
        X = keras.layers.Dense(nodes, activation="relu")(X)

    decoded = keras.layers.Dense(input_dims, activation="sigmoid")(X)

    decoder = keras.Model(inputs=encoded_input, outputs=decoded)

    auto_input = input_layer

    encoded_output = encoder(auto_input)

    decoded_output = decoder(encoded_output)

    auto = keras.Model(inputs=auto_input, outputs=decoded_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
