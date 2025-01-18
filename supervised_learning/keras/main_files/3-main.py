#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    labels = np.load('../../data/MNIST.npz')['Y_train'][:10]
    print(labels)
    print(one_hot(labels)) 
