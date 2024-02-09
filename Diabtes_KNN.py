#Diabtes predictor casestudy using K-Nearest-Neighbors Classifier.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


print("___________Diabetes predictor using K nearest neighbour Classifier______")

diabetes = pd.read_csv("diabetes.csv")

print("Column of dataset")
print(diabetes.columns)

print("First 5 record of dataset")
print(diabetes.head())

print("Dimention of diabetes data:{}".format(diabetes.shape))

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:,diabetes.columns!=
'Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

training_accuracy = []
test_accuracy = []

# try n neighbours from 1 to 10
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    #build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    #record training set accuracy
    training_accuracy.append(knn.score(X_train,y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test,y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("knn_compare_model")
plt.show()

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train, y_train)

print("Accuracy on training set:{:.2f}".format(knn.score(X_train, y_train)))

print("Accuracy on testing set:{:.2f}".format(knn.score(X_test, y_test)))