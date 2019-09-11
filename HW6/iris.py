# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:13:16 2018

Intro to Analytics 501 HW6

@author: Yunjia
"""

#### Import Libraries ####
from sklearn import datasets
import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Iris data
iris = datasets.load_iris()

mydata = pd.DataFrame(np.concatenate((iris.data, np.array([iris.target]).T), axis = 1), columns=iris.feature_names+['target'])

# Seperate training and final validation data set
valueArray = mydata.values
X = valueArray[:,0:4]
Y = valueArray[:,4]
test_size = 0.20
seed = 7
X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

# setup 10-fold cross validation to estimate the accuracy of different models
num_folds = 10
num_instances = len(X_train)
scoring = 'accuracy'

# add each algorithm and its name to the model array
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model, add results to a results array
results = []
names = []

for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# make predictions on validation data
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validate)

print(accuracy_score(Y_validate, predictions))
print(confusion_matrix(Y_validate, predictions))
print(classification_report(Y_validate, predictions))