import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('mock\iris.data.csv') 

print(df.head())

x = df.iloc[:,:4]
y = df.iloc[:,-1]

print(x.head())
print(y.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=10)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))





 