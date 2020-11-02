# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:02:15 2020

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
from general_funs import *
# init params
item_num = 9
M = np.zeros((item_num, item_num))
for i in range(item_num - 1):
    # M[i, i] += 0.5
    M[i, i + 1] += 1
    M[i + 1, i] -= 1
M[0, 0] = 1
M[-1, -1] = -1
# init experiment
inputs = [0, 8]
X = np.zeros(item_num)
Y = X.astype(np.float)
Z = Y[inputs]
X[inputs] = 1
outputs = np.empty((0, Z.size))
ys = np.empty((0, item_num))
epochs = range(1000)
for t in epochs:
    temp = 1 - t / 1000
    dY = -np.dot(M, np.dot(M.T, Y) - X) * 0.1
    Y += dY #+ np.random.randn(item_num) / 1000 
    ys = np.append(ys, [Y], axis = 0)
    Z = softmax(Y[inputs], temp)[0]
    outputs = np.append(outputs, [Z], axis = 0)
# outputs = softmax(ys[:, inputs])

fig, ax = plt.subplots(2, 1)
for i in inputs:
    ax[0].plot(epochs, ys[:, i], label = 'Y[{}]'.format(i))
for i in range(Z.size):
    ax[1].plot(epochs, outputs[:, i], label = 'output[{}]'.format(i))
ax[0].legend(loc = 'upper right')


