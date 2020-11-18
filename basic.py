# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:07:52 2020

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
from funs import *
# init params
x_num, y_num = 500, 100
item_num = 3
X = np.random.randint(2, size = (item_num, x_num))
Y = np.random.randint(2, size = (item_num, y_num))
Xfin, Yfin = feedback_inhibition_network('d'), feedback_inhibition_network('d')
Xfin.setup_matrix(X), Yfin.setup_matrix(Y)

epoch = 1000
key1 = Xfin.recognise(X[0], epoch)
outputs1 = np.dot(key1, Y)
key2 = Yfin.recognise(Y[0], epoch)
outputs2 = np.dot(key2, X)

diff1x = np.max(np.abs(key1 - [1,0,0]), axis = 1)
diff2y = np.max(np.abs(key2 - [1,0,0]), axis = 1)
diff1y = np.max(np.abs(outputs1 - Y[0]), axis = 1)
diff2x = np.max(np.abs(outputs2 - X[0]), axis = 1)

epochs = range(epoch + 1)
fig, ax = plt.subplots(1, 2)
ax[0].plot(epochs[1:], diff1x[1:])
ax[1].plot(epochs[1:], diff2y[1:])
ax[0].plot(epochs[1:], diff1y[1:])
ax[1].plot(epochs[1:], diff2x[1:])
ax[0].set_xscale('log')
ax[1].set_xscale('log')

