#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block

if __name__ == '__main__':
    X = K.Input(shape=(224, 224, 3))
    Y = inception_block(X, [64, 96, 128, 16, 32, 32])
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
