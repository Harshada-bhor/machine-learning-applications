##Play predictor casestudy using K-Nearest-Neighbors Classifier.

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def PlayPredictor(data_path):

    # step 1 Load data
    data = pd.read_csv(data_path,index_col=0) # data read serially index of 1st column and index of 2nd column

    print("Size of Actual dataset",len(data))

    # step 2 clean , prepare, and manipulate data
    feature_names = ["Whether","Temprature"]

    print("Names of features",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    #converting string labels into numbers
    weather_encoded = le.fit_transform(whether)
    print("weather",weather_encoded)

    #converting string labels into numbers
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)

    print("temparature",temp_encoded)

    # combining weather and temp into single list of tuple
    features = list(zip(weather_encoded,temp_encoded))

    # step 3 train the data
    model = KNeighborsClassifier(n_neighbors=3)

    # train the model using the training data
    model.fit(features,label)

    # step 4 Test data
    predicted = model.predict([[0,2]])# 0: overcast 2: mild
    print("predicted",predicted)

def main():

    print("Machine Learning Application")
    print("Play predictor application using K Nearest Knighbor algorithm")

    PlayPredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()


