import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans 

df = pd.read_csv("Mall_Customers.csv")
df = df.drop(["Gender", "Age"], axis = 1)
data = np.array(df)

k = 3
km = KMeans(n_clusters = k)
km.fit(data)

labels = km.labels_

print(labels)

