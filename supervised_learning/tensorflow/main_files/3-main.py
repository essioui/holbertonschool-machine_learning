#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
accuracy = calculate_accuracy(y, y_pred)
print(accuracy)
