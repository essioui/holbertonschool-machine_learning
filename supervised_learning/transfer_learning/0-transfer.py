#!/usr/bin/env python3
"""
Script to train a convolutional neural network to classify the CIFAR 10 dataset using MobileNetV2
"""
from tensorflow.keras.layers import Dense, Lambda, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

def preprocess_data(X, Y):
    """
    Pre-processes the data for the model

    parameters:
        X [numpy.ndarray of shape (m, 32, 32, 3)]:
            contains the CIFAR 10 data where m is the number of data points
        Y [numpy.ndarray of shape (m,)]:
            contains the CIFAR 10 labels for X

    returns:
        X_p: a numpy.ndarray containing the preprocessed X
        Y_p: a numpy.ndarray containing the preprocessed Y
    """
    # scale pixels between 0 and 1
    X_p = X.astype("float32") / 255.0
    Y_p = to_categorical(Y, 10) # One-Hot Encoding
    return X_p, Y_p

if __name__ == '__main__':
    """
    Trains a convolutional neural network to classify CIFAR 10 dataset using MobileNetV2
    Saves model to cifar10_mobilenetv2.h5
    """
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = keras.Input(shape=(32, 32, 3))

    # Resize images from (32, 32) to (224, 224)
    inputs_resized = keras.layers.Lambda(
        lambda x: tf.image.resize(x, (224, 224)))(inputs)

    # Load MobileNetV2 model without the top layers (include_top=False)
    base_model = keras.applications.MobileNetV2(include_top=False,
                                                  weights='imagenet',
                                                  input_shape=(224, 224, 3))

    activation = keras.activations.relu

    # Build the model using MobileNetV2 as a base
    X = base_model(inputs_resized, training=False)
    X = GlobalAveragePooling2D()(X)  # Global Average Pooling
    X = Dense(256, activation=activation)(X)
    X = Dropout(0.3)(X)
    outputs = Dense(10, activation='softmax')(X)

    model = Model(inputs=inputs, outputs=outputs)

    # Freeze the layers of MobileNetV2 to prevent training them
    base_model.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Early stopping and learning rate reduction callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)

    # Train the model
    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=64,
                        epochs=10, verbose=True, shuffle=True,
                        callbacks=[early_stopping, reduce_lr])

    # Save the trained model
    model.save('cifar10.h5')
