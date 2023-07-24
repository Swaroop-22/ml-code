# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:45:02 2021

@author: Swaroop Honrao
"""

#Import library
import pandas as pd

#import dataset
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#split datset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#train data on logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#predict result
y_pred = classifier.predict(X_test)

#make confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#compute accuracy with k-fold cross validation technique
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10)
print("Accuracy! {:.2f} %".format(accuracy.mean() * 100))
print("Std deviation! {:.2f} %".format(accuracy.std() * 100))
















