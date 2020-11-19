# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:42:21 2020

@author: anish
"""

import numpy as np
import matplotlib.pyplot as plt
from funs import *
# init params
x_num = 500
item_num = 10
M = np.full((item_num, x_num), 0.00)
for _ in range(item_num):
    M[_, _ * 50 : (_ + 1) * 50] = 0.5
Xfin = feedback_inhibition_network('d')
Xfin.setup_matrix(M)

X = np.zeros(x_num)
X[:70] = np.random.randint(2, size=70)

epoch = 100
key1 = Xfin.recognise(X, epoch)



epochs = range(epoch + 1)
fig, ax = plt.subplots(1, 2)
ax[0].plot(epochs[1:], key1[1:])

# response to band class