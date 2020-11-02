# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:07:52 2020

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
# init params
item_num = 8
M = np.zeros((item_num, item_num))
for i in range(item_num - 1):
    M[i, i] = 1
    M[i, i + 1] = 1

# init experiment
inputs = [4, 5]
X = np.zeros(item_num)
Y = X.astype(np.float)
X[inputs] = 1
outputs = np.empty((0, item_num))
epochs = range(100)
for t in epochs:
    dY = -np.dot(M, np.dot(M.T, Y) - X) * 0.1
    Y += dY
    outputs = np.append(outputs, [Y], axis = 0)

fig, ax = plt.subplots()
for i in range(item_num):
    ax.plot(epochs, outputs[:, i], label = i)
ax.legend()