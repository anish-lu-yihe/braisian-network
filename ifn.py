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
    
    def _dY_sub(self, X, Xhat, Y):
        return np.dot(self.M, X - Xhat)
    
    def _dY_div(self, X, Xhat, Y):
        numer = np.dot(self.M, np.divide(X, Xhat, out = np.ones_like(X), where = Xhat != 0) - 1)
        denom = np.sum(self.M, axis = 1)
        return np.multiply(Y, np.divide(numer, denom))
    
    def recognise(self, X, epoch = 1000, rate = 0.1):  
        _X = np.asarray(X, dtype = float)      
        _dY = {'div': self._dY_div, 'sub': self._dY_sub}[self.inhitype]
        _Y = np.dot(self.M, _X)
        Ys = [np.dot(self.M, _X)]
        for t in range(epoch):
            Xhat = np.dot(self.M.T, _Y)
            _Y += _dY(_X, Xhat, _Y) * rate
            Ys = np.concatenate((Ys, [_Y]), axis = 0)                
        return Ys
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        