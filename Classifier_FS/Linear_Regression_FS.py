import numpy as np
import pandas as pd

from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data 
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2) 

x_train = np.vstack((np.ones((x_train.shape[0],)), x_train.T)).T
x_test = np.vstack((np.ones((x_test.shape[0],)), x_test.T)).T

print(x_train.shape)
print(x_test.shape)

def Linear_Regression(x_train, y_train, learning_rate, iteration):
    m = y_train.size
    theta = np.zeros((x_train.shape[1],1))

    for i in range(iteration):
        y_pred = np.dot(x_train, theta)
        cost = ( 1/(2*m) ) * np.sum(np.square( y_pred - y_train ))
        d_theta = (1/m) * np.dot(x_train.T, y_pred - y_train )
        theta = theta - learning_rate * d_theta

    return theta

iteration = 10000
learning_rate = 0.000000005
theta = Linear_Regression(x_train, y_train, learning_rate, iteration)

y_pred = np.dot(x_test, theta)
print(y_pred)
error = ( 1/x_test.shape[0] ) * np.sum(np.abs( y_pred - y_test ))
print(error * 100)
print((1 - error) * 100 )


