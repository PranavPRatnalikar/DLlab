import pandas as pd
import numpy as np

df = pd.read_csv('mock/tennis.csv')
print(df.head())
print(df.describe())

from sklearn.preprocessing import LabelEncoder

df['outlook'] = LabelEncoder().fit_transform(df['outlook'])
df['temp'] = LabelEncoder().fit_transform(df['temp'])
df['humidity'] = LabelEncoder().fit_transform(df['humidity'])
df['windy'] = LabelEncoder().fit_transform(df['windy'])

x = df.drop(['play'],axis=1)
y = df['play']

print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.30,random_state=10)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train,y_train)

y_pred = gnb.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred)*100,"%")
