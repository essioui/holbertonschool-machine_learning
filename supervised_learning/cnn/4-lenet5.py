#!/usr/bin/env python3
"""
Module LeNet-5 (Tensorflow 1)
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow
    Args:
        x is a tf.placeholder of shape (m, 28, 28, 1)
            m is the number of images
        y is a tf.placeholder of shape (m, 10)
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes
    he_normal:
        tf.keras.initializers.VarianceScaling(scale=2.0)
    Returns:
        tensor for the softmax activated output
        training operation that utilizes Adam optimization
        tensor for the loss of the netowrk
        tensor for the accuracy of the network
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(pool1)

    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = tf.layers.Flatten()(pool2)

    fully_conneccted_1 = tf.layers.Dense(units=120,
                                         activation=tf.nn.relu,
                                         kernel_initializer=initializer)(
                                             flatten)

    fully_connected_2 = tf.layers.Dense(units=84,
                                        activation=tf.nn.relu,
                                        kernel_initializer=initializer)(
                                            fully_conneccted_1)

    logits = tf.layers.Dense(units=10,
                             kernel_initializer=initializer)(fully_connected_2)
    y_pred = tf.nn.softmax(logits)

    loss_function = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                                    logits=logits)

    train_op = tf.train.AdamOptimizer().minimize(loss_function)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred, train_op, loss_function, accuracy
