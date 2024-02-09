#  Titanic casestudy Using Logistic Regression.


import math
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def TitanicLogistic():

    # step 1 Load data
    titanic_data = pd.read_csv("MarvellousTitanicDataset.csv")

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passanger are" +str(len(titanic_data)))

    #Analyse the data
    print("Visualization: Survived and non survived passangers")
    figure()
    target = "Survived"

    countplot(data=titanic_data,x=target).set_title("Marvellous Infosystem: Survive and non survived passanger ")
    show()

    print("Visualization: Survived and non survived passangers based on gender")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target, hue="Sex").set_title("Marvellous Infosystem: Survive and non survived passanger based on Gender")
    show()

    print("Visualization: Survived and non survived passangers based on Passanger class")
    figure()
    target = "Survived"

    countplot(data=titanic_data, x=target, hue="Pclass").set_title(
        "Marvellous Infosystem: Survive and non survived passanger based on Passanger class ")
    show()

    print("Visualization: Survived and non survived passangers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystem: Survived and non survived passangers based on Age")
    show()

    print("Visualization: Survived and non survived passangers based on Passanger Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title(
        "Marvellous Infosystem: Survived and non survived passangers based on Fare")
    show()

    # cleaning the data
    titanic_data.drop("zero",axis=1,inplace = True)

    print("First 5 entries from loaded dataset after removing zero colomn")
    print(titanic_data.head(5))

    print("values of sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("values of sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first = True)
    print(Sex.head(5))

    print("values of Pclass column after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    print(Pclass.head(5))

    print("values of dataset after concatenating new columns")
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis = 1)
    print(titanic_data.head(5))

    print("values of dataset after removing irrelevent columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"], axis = 1,inplace = True)
    print(titanic_data.head(5))

    x = titanic_data.drop("Survived",axis=1)
    y = titanic_data["Survived"]

    # Data training
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.5)

    logmodel = LogisticRegression()

    logmodel.fit(xtrain,ytrain)

    # Data testing
    prediction = logmodel.predict(xtest)

    # calculate Accuracy
    print("Classification report of Logistic Regression is :")
    print(classification_report(ytest,prediction))

    print("confusion matrix of logistic Regression is :")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of logistic Regression is:")
    print(accuracy_score(ytest,prediction))


def main():

    print("Machine Learning Application")
    print("Logistic Regression on titanic set")

    TitanicLogistic()

if __name__ == "__main__":
    main()


