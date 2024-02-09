# this is iris casestudy
# trained and test model using Ensemble learning.


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()
x = iris['data']
y = iris['target']

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42) # 70% training and 30% test

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()

Vot_clf = VotingClassifier(estimators=[('lr',log_clf),('rnd',rnd_clf),('knn',knn_clf)], voting = 'hard')

Vot_clf.fit(x_train,y_train)

pred = Vot_clf.predict(x_test)

print("Testing accuracy is :",accuracy_score(y_test,pred)*100)



