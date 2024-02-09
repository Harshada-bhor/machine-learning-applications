
# iris casestudy using User Defined K-Nearest-Neighbors Classifier.


from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance


def euc(a,b):
    return distance.euclidean(a,b)

class KNeighborsClassifier:
    def fit(self,trainingdata, trainingtarget):
        self.TrainingData = trainingdata
        self.TrainingTarget = trainingtarget

    def closest(self,row):
        minimumdistance=euc(row,self.TrainingData[0])
        minimumindex= 0

        for i in range(1,len(self.TrainingData)):
            Distance = euc(row, self.TrainingData[i])
            if Distance < minimumdistance:
                minimumdistance = Distance
                minimumindex = i

        return self.TrainingTarget[minimumindex]

    def predict(self,TestData):
        predictions = []
        for value in TestData:
            result = self.closest(value)
            predictions.append(result)

        return predictions
def ML():
    Dataset = load_iris()   # 1 Load the data

    Data = Dataset.data
    Target= Dataset.target
# 2manipulates the data
    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size = 0.5)

    Classifier = KNeighborsClassifier()   #algorithm choosing
    
#3built the model
    Classifier.fit(Data_train,Target_train)
    
#4test the model
    Predictions = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test,Predictions)

    return Accuracy
#5 improve missing

def main():
    Ret= ML()

    print("Accuracy of this iris dataset with KNN is ", Ret*100)


if __name__== "__main__":
    main()


