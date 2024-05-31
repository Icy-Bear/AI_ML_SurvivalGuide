import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data 
Y = iris.target

x_train, x_test, y_trian, y_test = train_test_split(X, Y, test_size = 0.2) 




