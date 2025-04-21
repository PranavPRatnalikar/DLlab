import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

(xtrain,ytrain),(xtest,ytest) = tf.keras.datasets.mnist.load_data()


xtrain = xtrain/255.0
ytrain = ytrain/255.0

xtrain_cat = to_categorical(xtrain)
ytrain_cat = to_categorical(ytrain)


model = Sequential([
    
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation'relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(10,activation='softmax') 
])
