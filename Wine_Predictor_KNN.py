# Wine predictor casestudy using K-Nearest-Neighbors Classifier.

from sklearn import metrics
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def WinePredictor():

    # step 1 Load data
    wine = datasets.load_wine()


    # print name of features
    print(wine.feature_names)

    # label species(class_0,class_1,class_2)
    print(wine.target_names)

    #wine data top 5 records
    print(wine.data[0:5])

    #wine label (1:class_0,2:class_1,3:class_2)
    print(wine.target)


    Data_train, Data_test, Target_train, Target_test = train_test_split(wine.data, wine.target, test_size=0.3)

    Classifier = KNeighborsClassifier(n_neighbors=3)  # algorithm choosing

    # 3built the model
    Classifier.fit(Data_train, Target_train)

    # 4test the model
    Predictions = Classifier.predict(Data_test)

    Accuracy = metrics.accuracy_score(Target_test, Predictions)

    return Accuracy


def main():

    print("Machine Learning Application")
    print("Wine predictor application using K Nearest Knighbor algorithm")

    Ret = WinePredictor()
    print("Accuracy of this Wine dataset with KNN is ", Ret * 100, "%")


if __name__ == "__main__":
    main()


