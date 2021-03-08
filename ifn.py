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
        self.M = np.asarray(M, dtype = float)
        self.num_Y, self.num_X = self.M.shape
        
    def _dY_sub(self, X, Y, V, epsilon):
        E = X - np.dot(self.M.T, Y)
        dY = np.dot(self.M, E)
        return dY
    
    def _dY_div(self, X, Y, V, epsilon):
        Xe = np.dot(self.M.T, Y)
        Xe[Xe == 0] = epsilon
        E = np.divide(X, Xe)
        dY = np.multiply(Y, np.divide(np.dot(self.M, E), V) - 1)
        return dY

    def recognise(self, X, epoch = 1000, rate = 0.1, epsilon = 0.001):
        _dY = {'div': self._dY_div, 'sub': self._dY_sub}[self.inhitype]        
        V = np.sum(self.M, axis = 1) # element of V can never be 0, unless a memory contains all 0s.
        Y = np.dot(self.M, X)
        outputs = [np.dot(self.M, X)]
        for t in range(epoch):
            dY = _dY(X, Y, V, epsilon)
            # noise = np.random.normal(scale = epsilon, size = self.num_Y)
            noise = 0
            Y += dY * rate + noise
            outputs = np.concatenate((outputs, [Y]), axis = 0)
        return outputs
    
    def recognise2(self, X, epoch = 1000, rate = 0.1, epsilon = 0.001):        
        _dY = {'div': self._dY_div2, 'sub': self._dY_sub2}[self.inhitype]  
        Y = np.dot(self.M, X)
        outputs = [np.dot(self.M, X)]
        for t in range(epoch):
            Xhat = np.dot(self.M.T, Y)
            dY = _dY(X, Xhat, Y)
            Y += dY * rate
            outputs = np.concatenate((outputs, [Y]), axis = 0)
        return outputs
        
    def _dY_sub2(self, X, Xhat, Y):
        return np.dot(self.M, X - Xhat)
    
    def _dY_div2(self, X, Xhat, Y):
        Xhat[Xhat == 0] = 0.001
        numer = np.dot(self.M, np.divide(X, Xhat) - 1)
        denom = np.sum(self.M, axis = 1)
        return np.multiply(Y, np.divide(numer, denom))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        