#!/usr/bin/env python3
"""
This modules builds, trains, and saves a neural network classifier
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    Args:
        X_train is a numpy.ndarray containing the training input data
        Y_train is a numpy.ndarray containing the training labels
        X_valid is a numpy.ndarray containing the validation input data
        Y_valid is a numpy.ndarray containing the validation labels
        layer_sizes list containing the number of nodes in each layer
        activations list containing the activation functions for each layer
        alpha is the learning rate
        iterations is the number of iterations to train over
        save_path designates where to save the model
    Returns:
        the path where the model was saved
    """
    # Initialize placeholders and build the network
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add relevant components to the graph's collections
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Start training session
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            # Train the model on the training data
            feed_dict_train = {x: X_train, y: Y_train}
            feed_dict_valid = {x: X_valid, y: Y_valid}

            _, train_cost, train_accuracy = sess.run(
                [train_op, loss, accuracy], feed_dict=feed_dict_train)
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict=feed_dict_valid)

            # Print progress
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        tf.set_random_seed(0)
        # Save the model
        saver = tf.train.Saver()
        saved_path = saver.save(sess, save_path)

    return saved_path
