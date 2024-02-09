# Iris casestudy Using Clustering Kmeans Classifier.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


dataset = pd.read_csv("iris.csv")
x = dataset.iloc[:,[0,1,2,3]].values

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("The elbow method")
plt.xlabel("number of clusters")
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(n_clusters = 3,init = 'k-means++', max_iter = 300 ,
                n_init = 10,random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1],s=100,c='red',label = 'Iris-sentosa')

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1],s=100,c='blue',label = 'Iris-versicolor')

plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1],s=100,c='green',label = 'Iris-verginica')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 100,c= "yellow"
            ,label="Centroids")

plt.legend()

plt.show()

