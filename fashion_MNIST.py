#imports 
from keras.datasets import fashion_mnist
import numpy as np 
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from sklearn.metrics import classification_report

#import the MNIST fashion data 
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

#getting a visualization of the data 
single_img=x_train[0]
plt.imshow(single_img)
#corresponding category
print(y_train[0])
#
#
###Preprocessing the Data
#
#Normalizing x data 
x_train=x_train/255
x_test=x_test
#reshape x arrays
print(x_test.shape)
x_test=x_test.reshape(10000,28,28,1)
print(x_test.shape)
print(x_train.shape)
x_train=x_train.reshape(60000,28,28,1)
print(x_train.shape)

#apply one hot encoding to the y data 
y_cat_test=to_categorical(y_test)
y_cat_train=to_categorical(y_train)

##
##Building the model
##
#convolution layer 
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
#pooling layer
model.add(MaxPool2D(pool_size=(2,2)))

#Flatten images from 28 by 28 
model.add(Flatten())
#Dense layer 1
model.add(Dense(128,activation='relu'))
#Dense layer 2
model.add(Dense(10,activation='softmax'))
#compile model
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#overview of the model
model.summary()

#traning model
model.fit(x_train,y_cat_train,epochs=10)
#evaluating model
model.metrics_names
model.evaluate(x_test,y_cat_test)
predictions=model.predict_classes(x_test)
print(classification_report(y_test,predictions))