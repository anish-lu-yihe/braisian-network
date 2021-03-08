# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:06:47 2020

@author: Yihe Lu
"""
import numpy as np

class inhibitory_feedback_network:
    def __init__(self, M, inhibition_type = 'divisive'):
        self.reset_memory(M)
        if inhibition_type in ['s', 'sub', 'subtract', 'subtractive']:
            self.inhitype = 'sub'
        else:
            self.inhitype = 'div'

    def reset_memory(self, M):
        self.M = np.asarray(M)
        self.num_Y, self.num_X = self.M.shape
        
    def _dY_sub(self, X, Y, epsilon):
        E = X - np.dot(self.M.T, Y)
        dY = np.dot(self.M, E)
        return dY
    
    def _dY_div(self, X, Y, epsilon):
        Xe = np.dot(self.M.T, Y).astype(np.float)
        Xe[Xe == 0] = epsilon
        E = np.divide(X, Xe)
        dY = np.multiply(Y, np.divide(np.dot(self.M, E), self.V) - 1)
        return dY

    def recognise(self, X, epoch = 1000, rate = 0.1, epsilon = 0.001):
        _dY = {'div': self._dY_div, 'sub': self._dY_sub}[self.inhitype]        
        self.V = np.sum(self.M, axis = 1).astype(np.float)
        self.V[self.V == 0] = epsilon        
        Y = np.zeros(self.num_Y)
        outputs = [np.dot(self.M, X)]
        for t in range(epoch):
            dY = _dY(X, Y, epsilon)
            noise = np.random.normal(scale = epsilon, size = self.num_Y)
            Y += dY * rate + noise
            outputs = np.append(outputs, [Y], axis = 0)
        return outputs