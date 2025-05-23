#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block

if __name__ == '__main__':
    X = K.Input(shape=(224, 224, 256))
    Y = identity_block(X, [64, 64, 256])
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
