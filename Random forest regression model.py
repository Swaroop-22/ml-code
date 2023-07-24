# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:19:07 2021

@author: Swaroop Honrao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#training decision tree on whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 10, random_state=0)
regressor.fit(x,y)

#predicting new results
regressor.predict([[6.5]])

#visualising decision tree for higher resolution
x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Turth or Bluff(Random Forest Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()