# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:46:07 2021

@author: Swaroop Honrao
"""

#Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)    #To specify that there are no header in dataset
transactions = []
for i in range(0,7501):        #This for loop is for all rows in dataset
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])    #Sq brackets used to make transactions list of products
    #All the data in dataset must be string to used apriori model otherwise it won't work.

#Training Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003,
                min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2) 
  #Min value is 3 multiply by 7 days which will be divided by total rows
  #i.e. 3*7/7501 = 0.0027 ~ 0.003

#Displaying the first results coming directly from the output of the apriori function
results = list(rules)

#Putting the results well organised into pandas dataframe
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs,rhs,supports))
resultsinDF = pd.DataFrame(inspect(results), columns = ['Product 1','Produt 2', 'Support'])

#Displaying the result sorted by descending lifts
resultsinDF.nlargest(n = 10, columns='Support')
