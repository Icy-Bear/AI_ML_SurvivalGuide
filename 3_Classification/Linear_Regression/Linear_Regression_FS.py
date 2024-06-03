import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
iris = load_iris()
X = iris.data
Y = iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# One-hot encode the target values
encoder = OneHotEncoder()
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_regression(x_train, y_train, learning_rate, iteration):
    m, n = x_train.shape
    k = y_train.shape[1]
    theta = np.zeros((n, k))
    for i in range(iteration):
        logits = np.dot(x_train, theta)
        y_pred = softmax(logits)
        error = y_pred - y_train
        grad = np.dot(x_train.T, error) / m
        theta -= learning_rate * grad
    return theta

# Hyperparameters
iteration = 10000
learning_rate = 0.1

# Train the model
theta = softmax_regression(x_train, y_train_onehot, learning_rate, iteration)

# Predict
logits = np.dot(x_test, theta)
y_pred_proba = softmax(logits)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate the accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

