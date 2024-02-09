# Diabetes Casestudy Using Logistic Regression .


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)
print("___________Diabetes predictor using Logistic Regression______")

diabetes = pd.read_csv("diabetes.csv")

print("Column of dataset")
print(diabetes.columns)

print("First 5 record of dataset")
print(diabetes.head())

print("Dimention of diabetes data:{}".format(diabetes.shape))

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:,diabetes.columns!=
'Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

logreg= LogisticRegression()
logreg.fit(X_train,y_train)
print("Accuracy on training set:{:.3f}".format(logreg.score(X_train,y_train)))
print("Accuracy on testing set:{:.3f}".format(logreg.score(X_test,y_test)))

logreg001 = LogisticRegression(C=0.01)
logreg001.fit(X_train,y_train)
print("Accuracy on training set:{:.3f}".format(logreg001.score(X_train,y_train)))
print("Accuracy on testing set:{:.3f}".format(logreg001.score(X_test,y_test)))

