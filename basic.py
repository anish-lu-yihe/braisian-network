# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:07:52 2020

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
from funs import *
# init params
x_num, y_num = 10, 5
item_num = 3
X = np.random.randint(2, size = (item_num, x_num))
Y = np.random.randint(2, size = (item_num, y_num))
Xfin, Xbin = feedback_inhibition_network('d'), feedback_inhibition_network('d')
Yfin, Ybin = feedback_inhibition_network('d'), feedback_inhibition_network('d')
Xfin.setup_matrix(X), Xbin.setup_matrix(X.T)
Yfin.setup_matrix(Y), Ybin.setup_matrix(Y.T)
key1 = Xfin.recognise(X[0])[-1]
outputs1 = np.dot(Y.T, key1)
key2 = Yfin.recognise(Y[0])[-1]
outputs2 = np.dot(X.T, key2)

print(key1, outputs1, Y[0])
print(key2, outputs2, X[0])

# fig, ax = plt.subplots(1, 2)
# for i in range(y_num):
#     ax[0].plot(epochs, outputs1[:, i], label = i)
#     ax[1].plot(epochs, outputs2[:, i], label = i)
# ax[0].legend()
# # ax[1].legend()

