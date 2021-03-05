# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:00:21 2020

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
M = np.zeros((y_num, x_num))
for x, y in zip(X, Y):
    M += np.outer(y, x)

Xfin, Yfin = feedback_inhibition_network('d'), feedback_inhibition_network('d')
Xfin.setup_matrix(M), Yfin.setup_matrix(M.T)

epoch = 1000
key1 = Xfin.recognise(X[0], epoch)




