# this is Diabtes case study.
# using Random forest Classifier.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)
print("___________Python Automation & Machine learning__________")
print("___________Diabetes predictor using Randon Forest Classifier_________")

diabetes = pd.read_csv("diabetes.csv")

print("Column of dataset")
print(diabetes.columns)

print("First 5 record of dataset")
print(diabetes.head())

print("Dimention of diabetes data:{}".format(diabetes.shape))

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:,diabetes.columns!=
'Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

rf= RandomForestClassifier(n_estimators=100,random_state=0)
rf.fit(X_train,y_train)
print("Accuracy on training set:{:.3f}".format(rf.score(X_train,y_train)))
print("Accuracy on testing set:{:.3f}".format(rf.score(X_test,y_test)))

rf1 = RandomForestClassifier(max_depth=3,n_estimators=100,random_state=0)
rf1.fit(X_train,y_train)
print("Accuracy on training set:{:.3f}".format(rf1.score(X_train,y_train)))
print("Accuracy on testing set:{:.3f}".format(rf1.score(X_test,y_test)))

#print("Feature Importance:\n{}".format(rf.feature_importances_))

def plot_feature_importance_diabetes(model):
    plt.figure(figsize=(8,6))
    n_feature = 8
    plt.barh(range(n_feature),model.feature_importances_,align='center')
    diabetes_feature = [x for i,x in enumerate(diabetes.columns) if i!=8]
    plt.yticks(np.arange(n_feature),diabetes_feature)
    plt.xlabel("feature Importance")
    plt.ylabel("feature")
    plt.ylim(-1,n_feature)
    plt.show()

plot_feature_importance_diabetes(rf)