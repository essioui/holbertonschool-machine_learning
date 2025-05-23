#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
LSTMCell = __import__('3-lstm_cell').LSTMCell

np.random.seed(3)
lstm_cell = LSTMCell(10, 15, 5)
print("Wf:", lstm_cell.Wf)
print("Wu:", lstm_cell.Wu)
print("Wc:", lstm_cell.Wc)
print("Wo:", lstm_cell.Wo)
print("Wy:", lstm_cell.Wy)
print("bf:", lstm_cell.bf)
print("bu:", lstm_cell.bu)
print("bc:", lstm_cell.bc)
print("bo:", lstm_cell.bo)
print("by:", lstm_cell.by)
lstm_cell.bf = np.random.randn(1, 15)
lstm_cell.bu = np.random.randn(1, 15)
lstm_cell.bc = np.random.randn(1, 15)
lstm_cell.bo = np.random.randn(1, 15)
lstm_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
c_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, c, y = lstm_cell.forward(h_prev, c_prev, x_t)
print(h.shape)
print(h)
print(c.shape)
print(c)
print(y.shape)
print(y)
