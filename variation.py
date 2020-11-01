# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:02:15 2020

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
from general_funs import *
# init params
item_num = 8
M = np.zeros((item_num, item_num))
for i in range(item_num - 1):
    M[i, i] = 1
    M[i, i + 1] = 1

# init experiment
inputs = [2, 3]
X = np.zeros(item_num)
X[inputs] = .5
Y = X.astype(np.float)
outputs = np.empty((0, item_num))
epochs = range(1000)
for t in epochs:
    dY = -np.dot(M, np.dot(M.T, Y) - X) * 0.1
    Y += dY + np.random.randn(item_num) / 100
    # Z = np.multiply(softmax(Y), X)
    # Z = Z / np.sum(Z)
    Z = softmax(Y)
    dZ = Z - Y
    # print(dZ)
    Y -= dY
    outputs = np.append(outputs, [Y], axis = 0)

fig, ax = plt.subplots()
for i in range(item_num):
    ax.plot(epochs, outputs[:, i], label = i)
ax.legend()


