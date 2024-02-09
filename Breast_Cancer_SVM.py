
# Breast cancer case study with support vector machine

###############################
#Required python packages
###############################
from sklearn.metrics import accuracy_score
from sklearn import datasets 
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def SVM():
    #Load dataset
    cancer = datasets.load_breast_cancer()

    # print name of 13 features
    print("Features of the cancer datasets:",cancer.feature_names)

    # print the label type of cancer('malignant'n'benign')
    print("labels of the cancer datasets:", cancer.target_names)

    # print data (feature) shape
    print("shape of the cancer datasets:", cancer.data.shape)

    # print cancer data features(top 5 records)
    print("First 5 records are :", cancer.data[0:5])

    # print cancer labels(0:malignant,1:benign)
    print("Target of dataset:",cancer.target)

    # Split dataset into training set and test set
    X_train, X_test, y_train,y_test = train_test_split(cancer.data,cancer.target,
    test_size=0.3,random_state=109) # 70% training and 30% test

    # create a svm classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernal

    # test the model using the training set
    clf.fit(X_train, y_train)

    # predict responce of the datasets
    y_pred = clf.predict(X_test)

    # model Accuracy
    print("Accuracy of the model:",accuracy_score(y_test, y_pred)*100)

def main():
    print("___________Breast Cancer Support vector machine algorithm__________")

    SVM()


if __name__ == "__main__":
    main()
    

    
    
