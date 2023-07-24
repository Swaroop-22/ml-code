# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:07:17 2021

@author: Swaroop Honrao
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#training the linear regression model on whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#training polynomial regression model on whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


#visualising linear regression results
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Turth or Bluff(Linear Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#visualising poly regression results
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg_2.predict(x_poly), color = 'blue')
plt.title('Turth or Bluff(Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#visualising poly regression results(for smoth curve and higher resolution)
x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Turth or Bluff(Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with linear regression
lin_reg.predict([[6.5]]) #to create array we use double sq bracket because
                         #this data contain 2 dimension

#Predicting a new result with Polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))




