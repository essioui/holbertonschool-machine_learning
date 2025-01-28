#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':
    lib = np.load('../../data/labels_logits.npz')
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)
    confusion = create_confusion_matrix(labels, logits)
    print(confusion)
    np.savez_compressed('confusion.npz', confusion=confusion)
