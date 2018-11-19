# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#numpy is used for basic mathmatical operations
#panda will eat the data set i.e it is used to read the data set
#matapolt is used to plot graph
#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#to read all dataset add a colon ist colon is for row and another is for coloumn
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
# : represent the total values
# X denotes the independetnt  variable and y denotes the dependent variables

#spliting the dataset into test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)
#fiting the dataset into test and train dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#as the y points to the output so we put the preducted values in the y variable
#plotting graph
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training data)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Testing data)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
#blue is the prediction and red is the input

