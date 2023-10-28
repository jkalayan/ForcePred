#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#https://sites.google.com/view/ml-basics/linear-regression-and-gradient-descent
###LINEAR REGRESSION
###########################################################################
#y = b + wX, w = weight, b = bias, X = input, y = output
#best combination of bias and weight to get the lowest loss/cost function
X_and_y = [(0, 1), (1, 3), (2, 2), (3, 5), (4, 7), (5, 8), (6, 8), (7, 9), 
        (8, 10), (9, 12)]
data = pd.DataFrame(X_and_y, columns=['X', 'y'])
X = data['X']
y = data['y']
b = 0
w = 1.5
y_predicted = b + w * X
error = y - y_predicted
L2 = 0.5 * np.mean(error**2)
#print(L2)



###########################################################################
##manually guessing w values
w_guess = np.linspace(0.75, 2, num=20)
L2_list = []
for w in w_guess:
    y_predicted = b + w*X
    error = y - y_predicted
    L2 = 0.5 * np.mean(error**2)
    L2_list.append(L2)
#print(w_guess, L2_list)



###########################################################################
#Batch gradient decent optimisation to find w_best and b_best
#https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e
#w_best = w_guess - 1/k * (dL2/dw)_wrt_w_guess
#gradient of loss/error wrt w
#gradient of loss/error wrt b
b = 0
w = 1.5
epochs = 1
learning_rate = 0.01 #should be small (how large a step to take)
for epoch in range(epochs):
    y_predicted = b + w * X
    error = y - y_predicted
    L2 = 0.5 * np.mean(error**2)
    gradient_b = -np.mean(error)
    b = b - learning_rate * gradient_b
    gradient_w = -np.mean(error * X)
    w = w - learning_rate * gradient_w
    #if epoch%(epochs/10) == 0:
        #print(epoch, L2)
#print(b, w)


###########################################################################
#Generalising code to get y_predicted = b + w_1*X'_1 + w_2*X'_2 + ...
#let b = w_0*X'_0, where X'_0 = 1
#need to refactor the code to work in the general case
#use matrices and dot product @, which multiplies then adds up as in eq above
N = len(X)
ones = np.ones(N)
Xp = np.c_[ones,X] #adds a column of ones to X for the bias
y = y.values.reshape(1,-1) #turn y into a matrix
#print(Xp)
#w = np.array([[0, 1.5]])
w = 2*np.random.rand(2)-1 #get random values between -1 and 1
epochs = 1
learning_rate = 0.01 #should be small (how large a step to take)
for epoch in range(epochs):
    y_predicted = w @ Xp.T #transpose Xp to get right shape
    error = y - y_predicted
    L2 = 0.5 * np.mean(error**2)
    gradient = -(1/N) * error @ Xp #divide by N to get the average
    w = w - learning_rate * gradient
    #if epoch%(epochs/10) == 0:
        #print(epoch, L2)
#print(w)



###########################################################################
#multiple linear regression (with multiple X variables/features)
#https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv
data = pd.read_csv('delaney_processed.csv')
#print(list(data))
y = data['measured log solubility in mols per litre']
X = data[['Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors', 
        'Number of Rings', 'Number of Rotatable Bonds', 
        'Polar Surface Area']]
N = len(X) #number of datapoints
ones = np.ones(N)
Xp = np.c_[ones,X] #adds a column of ones to X for the bias
y = y.values.reshape(1,-1) #turn y into a matrix
np.random.seed(0)
w = 2*np.random.rand(Xp.shape[1])-1 #get random values between -1 and 1 for 
        #x number of features
epochs = 100_000
learning_rate = 0.00002 #should be small (how large a step to take), l
        #if larger gradient, need to take a smaller step
for epoch in range(epochs):
    y_predicted = w @ Xp.T #transpose Xp to get right shape
    error = y - y_predicted
    L2 = 0.5 * np.mean(error**2)
    gradient = -(1/N) * error @ Xp #divide by N to get the average
    w = w - learning_rate * gradient
    if epoch%(epochs/10) == 0:
        print(epoch, L2)
print(w)



#plt.scatter(X, y)
#plt.plot(X, y_predicted)







#https://sites.google.com/view/ml-basics/logistic-regression
#LOGISTIC REGRESSION
###########################################################################
#take linear regression code and add an activation function to it
#https://raw.githubusercontent.com/animesh-agarwal/Machine-Learning/master/LogisticRegression/data/marks.txt


