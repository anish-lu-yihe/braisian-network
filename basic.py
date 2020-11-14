# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:07:52 2020

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
# init params
x_num, y_num = 8, 7
M = np.zeros((y_num, x_num))
for i in range(y_num):
    M[i, i:i+2] = 1
# M[-1]=M[1]
# init experiment
inputs = [2,4]
epsilon = 0.01
rate = 0.1
X1, X2 = np.zeros(x_num), np.full(x_num, epsilon)
Y1, Y2 = np.zeros(y_num), np.full(y_num, epsilon)
X1[inputs], X2[inputs] = 1, 1
# X1 = [0,0,0.1,0.5,0.5,0.7,0,0]
# X2 = [epsilon,epsilon,0.1,0.5,0.5,0.7,epsilon,epsilon]
print(M), print(X1)
outputs1, outputs2 = np.empty((0, y_num)), np.empty((0, y_num))
epochs = range(100)
for t in epochs:
    dY1 = -np.dot(M, np.dot(M.T, Y1) - X1) + np.abs(np.random.normal(0, epsilon, y_num))
    Y1 += dY1 * rate
    outputs1 = np.append(outputs1, [Y1], axis = 0)
    
    xxx = np.dot(M.T, Y2)
    xxx[xxx == 0] = epsilon
    E = np.divide(X2, xxx)
    bracket = np.divide(np.dot(M, E), np.sum(M, axis = 1)) - 1
    dY2 = np.multiply(Y2, bracket) + np.abs(np.random.normal(0, epsilon, y_num))
    Y2 += dY2 * rate
    outputs2 = np.append(outputs2, [Y2], axis = 0)

fig, ax = plt.subplots(1, 2)
for i in range(y_num):
    ax[0].plot(epochs, outputs1[:, i], label = i)
    ax[1].plot(epochs, outputs2[:, i], label = i)
ax[0].legend()
# ax[1].legend()

