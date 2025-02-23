#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow import keras as K
projection_block = __import__('3-projection_block').projection_block

if __name__ == '__main__':
    X = K.Input(shape=(224, 224, 3))
    Y = projection_block(X, [64, 64, 256])
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
