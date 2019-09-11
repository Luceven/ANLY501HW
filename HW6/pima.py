# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:58:49 2018

ANLY 501 Part 2

@author: Yunjia
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
    
    
def model_compare(mydata, X, Y, test_size, seed, num_folds):
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    num_instances = len(X_train)
    scoring = 'accuracy'
    
    # add each algorithm and its name to the model array
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    
    print("Training Performance\n")
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
        
    print("\nTesting Performance\n")
    accuracy = []
    # make predictions on validation data
    for name, model in models:
        classifier = model
        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_validate)
        acc = accuracy_score(Y_validate, predictions)
        accuracy.append(acc)
    
    for name, acc in zip(names, accuracy):
        print("The ", name, " classifier: ", acc, "% accuracy")


def main():
    # Load Pima indian diabetes data
    col_names = ['num_pregnant', 'pla_tolerance_test', 'dbp', 'tkft', '2hsi', 'bmi', 'dpf', 'age', 'class']
    mydata = pd.read_csv('./Pima_Diabetes_Dataset.txt', sep=',', names=col_names)
    
    # plot the Pima data
    # plot histogram
    #mydata.hist(figsize=(10,8))
    
    # scatter plot matrix
    #pd.plotting.scatter_matrix(mydata, figsize=(20,20))
    
    # set test size, seed, and number of folds
    test_size = 0.25
    seed = 7
    # setup 10-fold cross validation to estimate the accuracy of different models
    num_folds = 10
    valueArray = mydata.values
    # Seperate training and final validation data set
    X = normalize(valueArray[:,0:8], norm='max')
    Y = valueArray[:,8]
    
    print("====Classification Model Performance Compare====\n")
    
    # model compare with normalized data
    model_compare(mydata, X, Y, test_size, seed, num_folds)
    
    selector_list = [chi2, f_classif, mutual_info_classif]
    
    for selector in selector_list:
        # select 4 best features using Chi-Square, ANOVA F, and Mutual information
        select_k = SelectKBest(score_func=selector, k=4)
        X_new = select_k.fit_transform(X, Y)
        # Get the col names of selected features
        #mask = select_k.get_support()
        #new_features = mydata.columns[mask]
        #names = mydata.columns.values[select_k.get_support()]
        
        print("\n\n====Model Performance Compare after Feature Selection Using ", selector, "====\n")
        #print("Features Selected are: ", names, "\n")
        
        # model compare after feature selection
        model_compare(mydata, X_new, Y, test_size, seed, num_folds)

if __name__ == "__main__":
    main()

#print(accuracy_score(Y_validate, predictions))
#print(confusion_matrix(Y_validate, predictions))
#print(classification_report(Y_validate, predictions))