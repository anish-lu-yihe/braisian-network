# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:06:47 2020

@author: Yihe Lu
"""
import numpy as np

class inhibitory_feedback_network:
    def __init__(self, inhibition_type = 'divisive'):
        if inhibition_type in ['s', 'sub', 'subtract', 'subtractive']:
            self.inhitype = 'sub'
        else:
            self.inhitype = 'div'

    def setup_matrix(self, M):
        self.M = np.asarray(M)
        self.num_Y, self.num_X = self.M.shape

    def recognise(self, X, epoch = 1000, rate = 0.1, epsilon = 0.001):
        Dxn_recog = {'div': self.__recognise_div,
                     'sub': self.__recognise_sub}
        return Dxn_recog[self.inhitype](X, epoch, rate, epsilon)
        
    def _recognise_sub(self, X, epoch, rate, epsilon):
        Y = np.zeros(self.num_Y)
        outputs = np.empty((0, self.num_Y))
        for t in range(epoch):
            E = X - np.dot(self.M.T, Y)
            dY = np.dot(self.M, E)
            noise = np.random.normal(scale = epsilon, size = self.num_Y)
            Y += dY * rate + noise
            outputs = np.append(outputs, [Y], axis = 0)
        return outputs
    
    def _recognise_div(self, X, epoch, rate, epsilon):
        Y = np.zeros(self.num_Y)
        V = np.sum(self.M, axis = 1).astype(np.float)
        V[V == 0] = epsilon
        outputs = [np.dot(self.M, X)]
        for t in range(epoch):
            Xe = np.dot(self.M.T, Y).astype(np.float)
            Xe[Xe == 0] = epsilon
            E = np.divide(X, Xe)
            dY = np.multiply(Y, np.divide(np.dot(self.M, E), V) - 1)
            noise = np.abs(np.random.normal(scale = epsilon, size = self.num_Y))
            Y += dY * rate + noise
            outputs = np.append(outputs, [Y], axis = 0)
        return outputs