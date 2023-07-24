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
y = y.reshape(len(y),1)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#Training SVR model on whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)

#Predicting a new results
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

#visualising SVR results
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Truth or bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising SVR results for higher resolution and smooth curve
x_grid = np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Truth or bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()