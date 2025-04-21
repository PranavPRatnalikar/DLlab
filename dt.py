import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.data.csv')
df.head()
df.describe()

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
x=df.iloc[:,:4]
y=df.iloc[:,-1]
xtrain,xtest,ytrain,ytest  = train_test_split(x,y,test_size=0.2)

dtc.fit(xtrain,ytrain)

from sklearn.metrics import accuracy_score
print("training acc :",accuracy_score(ytrain,dtc.predict(xtrain)))
print("testing acc :",accuracy_score(ytest,dtc.predict(xtest)))
