
# Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans



# Data Collection and Analysis

customer_data = pd.read_csv('Mall_Customers.csv')
customer_data.head()
customer_data.shape
customer_data.info()


# Checking for Missing Values

customer_data.isnull().sum()


# Choosing the annual income and spending score column

X = customer_data.iloc[:,[3,4]].values


print(X)

# Choosing the correct number of clusters

# Finding the Wcss value for different number of clusters


wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = 1, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)


# plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# Optimum number of clusters = 5

# Training the  k-means clustering model

kmeans = KMeans(n_clusters = 5, init ='k-means++', random_state = 0)

# return a label for eachbdata point based on thier clusters

y = kmeans.fit_predict(X)
print(y)


# Visualizing all Clusters

plt.figure(figsize = (8,8))
plt.scatter(X[y == 0,0], X[y == 0,1], s = 50, c='green', label = 'Cluster 1')
plt.scatter(X[y == 1,0], X[y == 1,1], s = 50, c='red', label = 'Cluster 2')
plt.scatter(X[y == 2,0], X[y == 2,1], s = 50, c='yellow', label = 'Cluster 3')
plt.scatter(X[y == 3,0], X[y == 3,1], s = 50, c='violet', label = 'Cluster 4')
plt.scatter(X[y == 4,0], X[y == 4,1], s = 50, c='blue', label = 'Cluster 5')

# plot the centroids

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
