# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:59:12 2021

@author: Swaroop Honrao
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#Splitting dataset into Training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Training multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting Test set result
y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)  #to print only 2 digits after decimal point
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
#concatenate is used to add two columns i.e. y_pred and y_test and reshape function is used to print horz rows to verticle
#and to print the concatetnate table verticle the value of axis should be 1 i.e. final value in above syntax


#making a single prediction(R&D spend = 160000, admin = 130000, marketing = 300000)
#country- california
print(regressor.predict([[1,0,0,160000,130000,300000]]))

#getting final linear regression model with values of coefficient
print(regressor.coef_)
print(regressor.intercept_)





