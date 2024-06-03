# importing important module
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

# loading iris dataset
iris = load_iris()
x = iris.data
y = iris.target

#spliting dataset in features and labels
train_x, test_x, train_y, test_y = train_test_split(x , y , test_size = 0.3 , random_state = 42)

# Standardizing the features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

# Model creation
svc = SVC(kernel = 'linear', C = 1.0 , random_state = 42 ) 

# training time
svc.fit(train_x,train_y)

# let's enter in the battle ground
y_pred = svc.predict(test_x)

# Evaluation 
print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))
