import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/nsidd/OneDrive/Desktop/Python Datasets/letter-challenge-unlabeled.csv')
X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values
#Viewing the letters
dataset['letter'].value_counts()
dataset.isnull().sum()

#Data Cleaning
dataset['letter']=dataset['letter'].str.replace('?','')  #Replacing ? with blanks
dataset['letter']=dataset['letter'].replace(r'\s+( +\.)|#',np.nan,regex=True).replace('',np.nan) #Replaces blanks with NaN
dataset['letter'].value_counts()
dataset['letter']=dataset['letter'].fillna('-')     # Filling Blanks with '-'
dataset['letter'].value_counts()

#Replacing the letters for better identification
dataset['letter']=dataset['letter'].str.replace('+','0')     #Replacing '+' with 0
dataset['letter']=dataset['letter'].str.replace('-','1')     #Replacing '-' with 1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

#Part2
#ANN 
#Import keras library 
import keras
from keras.models import Sequential
from keras.layers import Dense	

#Initialising ANN
classifier = Sequential()

#Adding Input Layer
classifier.add(Dense(output_dim=7,kernel_initializer = 'uniform',activation = 'relu',input_dim=16))
	
#Adding second Hidden layer
classifier.add(Dense(output_dim=7,kernel_initializer = 'uniform',activation = 'relu'))

#Adding output Layer
classifier.add(Dense(output_dim=1,kernel_initializer = 'uniform',activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to Training Set
classifier.fit(X_train,y_train, batch_size= 10, epochs=100)

#Part 3----Making Predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.7)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)






