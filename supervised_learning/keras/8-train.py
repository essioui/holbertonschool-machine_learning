#!/usr/bin/env python3
"""
Module defines update the function def train_model
to also save the best iteration of the model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """
    update the function def train_model
    to also save the best iteration of the model
    Args:
        save_best: boolean indicating to save the model
        filepath: file path where the model should be saved
    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    if early_stopping and validation_data:
        early_stopping_cb = K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_cb)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay_cb = K.callbacks.LearningRateScheduler(scheduler,
                                                        verbose=verbose)
        callbacks.append(lr_decay_cb)

    if save_best and filepath is not None:
        checkpoint_cb = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            verbose=verbose
        )
        callbacks.append(checkpoint_cb)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
