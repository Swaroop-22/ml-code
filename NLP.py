# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:12:40 2021

@author: Swaroop Honrao
"""

#Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)  #quoting is used to ignore double quotes("") in the text

#Cleaning the text
import re    #this library is used to simplify the reviews
import nltk  #this library allows us to download and symbol of stop words
nltk.download('stopwords')   # stop words are words which we don't want to include in our reviews after cleaning.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []    #Initailising an empty list
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  #sub fun can replace anything in text
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating bags of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
len(x[0])

#Split dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Training navie bayes model on training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

#predicting test set result
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_pred),1)), 1))

#Making Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
a = accuracy_score(y_test, y_pred)
print(a)
