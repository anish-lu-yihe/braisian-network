# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:07:52 2020

@author: anish
"""
import numpy as np

M = np.asarray([[1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0]])

Y = np.zeros(5)
Z = Y
X = np.asarray([0, 0, 1, 1, 0])

for t in range(100):
    dY = -np.dot(M, np.dot(M.T, Y) - X) * 0.1
    Y += dY
    print(Y)