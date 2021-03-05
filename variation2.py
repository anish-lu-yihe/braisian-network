
import numpy as np
import matplotlib.pyplot as plt
# init params
x_num, y_num = 8, 7
M = np.zeros((y_num, x_num))
for i in range(y_num):
    M[i, i:i+2] = 1

# init experiment
inputs = [1,6]
epsilon = 0.001
rate = 0.1
X1, X2 = np.zeros(x_num), np.full(x_num, epsilon)
Y1, Y2 = np.zeros(y_num), np.full(y_num, epsilon)
X1[inputs], X2[inputs] = 1, 1
X0 = list(X1)
epochs = range(100)
steps = range(20)
fig, ax = plt.subplots(1, 2)
outputs1, outputs2 = np.empty((0, y_num)), np.empty((0, y_num))
for _ in steps:
    for t in epochs:      
        
        dY1 = -np.dot(M, np.dot(M.T, Y1) - X1) 
        Y1 += dY1 * rate
        
        
        xxx = np.dot(M.T, Y2)
        xxx[xxx == 0] = epsilon
        E = np.divide(X2, xxx)
        bracket = np.divide(np.dot(M, E), np.sum(M, axis = 1)) - 1
        dY2 = np.multiply(Y2, bracket) + np.abs(np.random.normal(0, epsilon, y_num))
        Y2 += dY2 * rate
        
    outputs1 = np.append(outputs1, [Y1], axis = 0)
    outputs2 = np.append(outputs2, [Y2], axis = 0)



    
        
    
    X1[:y_num] = Y1    
    # X1 = np.multiply(X1, X0)
    # X1 = X1 / np.max(np.abs(X1))
    X2[:y_num] = Y2
    # X2 = np.multiply(X2, X0)
    X2 = X2 / np.sum(X2)

for i in range(y_num):
    ax[0].plot(steps, outputs1[:, i], label = i)
    ax[1].plot(steps, outputs2[:, i], label = i)
ax[-1].legend()

