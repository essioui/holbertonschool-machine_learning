#!/usr/bin/env python3
"""
This modules evaluates the output of a neural network
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network
    Args:
        X is a numpy.ndarray containing the input data to evaluate
        Y is a numpy.ndarray containing the one-hot labels for X
        save_path is the location to load the model from
    Returns:
        the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        # Load the saved model
        saver = tf.train.import_meta_graph(f"{save_path}.meta")
        saver.restore(sess, save_path)

        # Retrieve tensors from the collection
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        # Evaluate the model
        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        cost = sess.run(loss, feed_dict={x: X, y: Y})

    return pred, acc, cost
