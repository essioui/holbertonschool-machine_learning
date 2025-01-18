#!/usr/bin/env python3
"""
Module defines update the function def train_model
to also train the model using early stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Update function train
    Args:
        early_stopping: boolean that indicates whether early stopping
        patience: patience used for early stopping
    Returns:
        The History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stopping_cb = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_cb)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
