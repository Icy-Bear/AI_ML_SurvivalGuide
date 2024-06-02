# importing important modules
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# loading datasets
iris = load_iris()
x = iris.data
y = iris.target

# spliting the data into features and predictor variables
train_x , test_x , train_y , test_y = train_test_split(x , y , test_size = 0.2 , random_state = 42 )

# creation of model
LoR = LogisticRegression(max_iter = 100)

# Training Time
LoR.fit(train_x,train_y)

# Let's Fight 
y_pred = LoR.predict(test_x)

# Evaluation 
accuracy = accuracy_score(test_y,y_pred)
print(f"Accuracy : {accuracy}")



