#!/usr/bin/env python3
"""
Module defines analyze validaiton data
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    update the function 4-train.py to also analyze validaiton data
    Args:
        validation_data:
            data to be analyzed during model training
    Return:
        the History object generated after training the model
    """
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle
                          )
    return history
