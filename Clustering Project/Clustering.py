import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn import mixture

data = pd.read_csv('sshekha4.csv', header=None)

#General 3D visulazation of the data provided before starting with any of the clustering algorithm
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, 0], data.loc[:, 1], data.loc[:, 2], s=[1]*len(data))
plt.show()

# Task 1
fig = plt.figure(figsize=(5, 5))
fig.add_subplot(111)
z = linkage(data, 'ward')
dendrogram(z)
plt.title('Hierarchical Clustering')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
# Below are the horizontal lines of maximum seperation
plt.axhline(y=3730)
plt.axhline(y=5205)
# Below is a line intersecting the lines of maximum separation
# plt.axhline(y=4475)
plt.show()
# Hence, there are 3 clusters if the linkage criteria is chosen as Ward's criteria
hierarchicalcluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
hclabels = hierarchicalcluster.fit_predict(data)
# Plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, 0], data.loc[:, 1], data.loc[:, 2], c=hclabels, cmap='rainbow', marker='o', s=[1]*len(data)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

#Task 2
# First finding the Sum of Squared Errors for different values of k
sse = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sse[k] = kmeans.inertia_ # This gives the sum of squared distances of the data points to its closest centroid
plt.figure(figsize=(5,5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

## Experimental Section Starts

# #Since elbow method does not give a clear elbow, using AIC to determine the number of the clusters
# sse = {}
# aic = {}
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(data)
#     sse[k] = kmeans.inertia_ # This gives the sum of squared distances of the data point to its closest centroid
#     aic[k] = 2*3 - 2*math.log(sse[k]) # AIC= 2k - 2ln(sse) where k is the number of features
# plt.figure(figsize=(5,5))
# plt.plot(list(aic.keys()), list(aic.values()))
# plt.xlabel("Number of clusters")
# plt.ylabel("AIC")
# plt.show()

## Experimental Section Ends

# After finding the appropriate number of clusters, applying KMeans on the selected cluster size
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
klabels = kmeans.predict(data)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, 0], data.loc[:, 1], data.loc[:, 2], c=klabels, cmap='Set1', marker='o', s=[1]*len(data)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Task 3
def minptsEpsilonFinder(minpts):
    ptsradius = []
    for i in range(len(data)):
        dist = []
        for j in range(len(data)):
            dist.append(distance.euclidean(data.iloc[i], data.iloc[j]))
        dist.sort()
        ptsradius.append(dist[minpts-1])
    ptsradius.sort(reverse=True)
    # Plotting the above distances
    fig = plt.figure(figsize=(5,5))
    plt.plot(ptsradius)
    plt.xlabel("x")
    plt.ylabel("radius")
    plt.show()
    
# Now applying the DBScan algorithm
for i in range(3, 7):
    minptsEpsilonFinder(i)
    
# Since the curves for different minpts are very similar, taking Minpts = 4 and epsilon = 100
# Applying the DBScan Algorithm for epsilon = 100 as there is a bend at y = 100
dbscan = DBSCAN(eps=100, min_samples=4, metric='euclidean')
dbscanlabels = dbscan.fit_predict(data)
# print(dbscanlabels)
myset = set(dbscanlabels)
# print(myset)    
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, 0], data.loc[:, 1], data.loc[:, 2], c=dbscanlabels, cmap='rainbow', marker='o', s=[1]*len(data)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Extra Credit Task (Gaussian Mixture Decomposition)
# Finding the scores for the log likelihood instead of the likehood. They are essentially the same
mlscores = [] # keeps the values of the log likelihood 
def likelihoodfinder(k):
    gmm = GaussianMixture(n_components=k)
    gmm.fit(data)
    mlscores.append(gmm.score(data))
    
xlist = []
for i in range(2, 11):
    xlist.append(i)
for i in range(2, 11):
    likelihoodfinder(i)
fig = plt.figure(figsize=(5,5))
plt.xlabel("Number of distributions (k)")
plt.ylabel("likelihood")
plt.plot(xlist, mlscores)
fig.tight_layout()
plt.show()

# Since there is a sudden change in slope for k=3, after the which the likelihood increases very slowly, taking k=3
gmm = GaussianMixture(n_components=3)
gmm.fit(data)
# print(gmm.means_)
# print(gmm.score(data))
gmmlabels = gmm.predict(data)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.loc[:, 0], data.loc[:, 1], data.loc[:, 2], c=gmmlabels, cmap='rainbow', marker='o', s=[1]*len(data));
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()