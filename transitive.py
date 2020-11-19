# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:02:15 2020

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
from general_funs import *
from funs import *
# init params
item_num = 9
M = np.zeros((item_num, item_num))
for i in range(item_num - 1):
    M[i, i] += 1
    M[i, i + 1] += 1
    # M[i + 1, i] += -0.5
    # M[i + 1, i + 1] += -0.5

# init experiment
inputs = [3, 6]
X = np.zeros(item_num)
Y = X.astype(np.float)
Z = Y[inputs]
X[inputs] = 1
outputs = np.empty((0, Z.size))
ys = np.empty((0, item_num))
epoch_num = 100
epochs = range(epoch_num)
fin = feedback_inhibition_network()
fin.setup_matrix(M)

cc=0
for _ in range(100):
    ys = fin.recognise(X, epoch_num)
    if ys[-1, inputs[0]] > ys[-1, inputs[1]]:
        cc+= 1
print(cc/100)
# the behaviour of choosing from inference pairs is purely random

fig, ax = plt.subplots(1, 2)
for i in range(item_num):
    ax[0].plot(epochs, ys[1:, i], label = 'Y[{}]'.format(i))
# for i in range(Z.size):
#     ax[1].plot(epochs, outputs[:, i], label = 'output[{}]'.format(i))
ax[0].legend(loc = 'upper right')


