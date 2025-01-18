#!/usr/bin/env python3
"""
Module defines update the function def train_model
 to also train the model with learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Update the function train model
     to also train the model with learning rate decay
     Args:
        learning_rate_decay: boolean indicates whether learning rate decay
        alpha is the initial learning rate
        decay_rate is the decay rate
    Returns:
        The History object generated during training.
    """
    # Learning Rate Decay Scheduler
    def lr_scheduler(epoch):
        """Calculate the decayed learning rate."""
        new_lr = alpha / (1 + decay_rate * epoch)
        return new_lr

    callbacks = []

    # Add Early Stopping if enabled
    if early_stopping and validation_data:
        early_stopping_cb = K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_cb)

    # Add Learning Rate Decay if enabled
    if learning_rate_decay and validation_data:
        lr_decay_cb = K.callbacks.LearningRateScheduler(
            lr_scheduler, verbose=1
        )
        callbacks.append(lr_decay_cb)

    # Train the model
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
