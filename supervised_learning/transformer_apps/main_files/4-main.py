#!/usr/bin/env python3
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
import tensorflow as tf

tf.random.set_seed(0)
data = Dataset(32, 40)
for inputs, target in data.data_train.take(1):
    print(create_masks(inputs, target))
