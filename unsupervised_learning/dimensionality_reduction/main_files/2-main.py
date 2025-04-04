#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
pca = __import__('1-pca').pca
P_init = __import__('2-P_init').P_init

X = np.loadtxt("../data/mnist2500_X.txt")
X = pca(X, 50)
D, P, betas, H = P_init(X, 30.0)
print('X:', X.shape)
print(X)
print('D:', D.shape)
print(D.round(2))
print('P:', P.shape)
print(P)
print('betas:', betas.shape)
print(betas)
print('H:', H)
