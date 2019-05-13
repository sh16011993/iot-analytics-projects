import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook

data = pd.read_csv("sshekha4.csv", header=None)

# First copying the original data to a different variable
originalData = data

# Task 1
# Scatter Plot to show the clustering of the data set
# Using kmeans to find the number of clusters

# First finding the Sum of Squared Errors for different values of k to determine the number of clusters using the elbow method
sse = {}
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sse[k] = kmeans.inertia_ # This gives the sum of squared distances of the data points to its closest centroid
fig = plt.figure(figsize=(3,3))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
fig.tight_layout()
plt.show()
# The graph shows that the elbow is clearly visibe at k=2
# Applying k-means with number of clusters=2
kmeans = KMeans(n_clusters=2)
kmeans.fit(data.loc[:, 0:1])
klabels = kmeans.predict(data.loc[:, 0:1])
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.scatter(data.loc[:, 0], data.loc[:, 1], c=klabels, cmap='rainbow', marker='o', s=[3]*len(data)) 
ax.set_xlabel("X1")
ax.set_ylabel("X2")
plt.show()

# Task 2
# Scaling the values
data.loc[:, 0] = data.loc[:, 0].transform(lambda x : x/x.max())
data.loc[:, 1] = data.loc[:, 1].transform(lambda x : x/x.max())
# Scaled values
print(data.loc[:, 0:1])

# Task 3, 4, 5, 6, 7
# Using k-fold validation (k=5). Then, applying the SVM method with a penatly cost 
gammaCScores = []
def gammaCfinder(c, g):
    model = svm.SVC(kernel='rbf', C=c, gamma=g)
    scores = cross_val_score(model, data.values[:,0:2], data.values[:,2:3].ravel(), cv=5)  
    gammaCScores.append([c, g, scores.mean()])

#Below is the loose grid search
c_range = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
gamma_range = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]

for i in c_range:
    for j in gamma_range:
        gammaCfinder(i, j)
        
meanScores = []
maxgammaCScore = [-1,-1,-1]
for val in gammaCScores:
    meanScores.append(val[2])
    if(val[2] > maxgammaCScore[2]):
        maxgammaCScore = val
        
print("Maximum Accuracy (loose grid search): ", max(meanScores))
print("Maximum Accuracy (c, gamma, max. accuracy values) (loose grid search): ", maxgammaCScore)

# Converting the list of accuracies to sets to identify the regions with same accuracy
s = set(meanScores)

# There are 6 different (distinct) accuracies obtained
print("Distinct sets (loose grid search): ", s)
print("Length: ", len(s))

# Before proceeding with the fine grid search, taking a look at the plot for the loose grid search
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([val[0] for val in gammaCScores], [val[1] for val in gammaCScores], [val[2]*100 for val in gammaCScores], c=meanScores, cmap='Set1', marker='o', s=[5]*len(meanScores))
ax.set_xlabel("C")
ax.set_ylabel("gamma")
ax.set_zlabel("% Accuracy")
plt.show()

# There is only one point having a maximum accuracy of 94.12 %. Hence, doing a fine grid search in the neighborhood of that point

# Now, doing the fine grid search
fine_c_range = [2**-7, 2**-6.75, 2**-6.5, 2**-6.25, 2**-6, 2**-5.75, 2**-5.5, 2**-5.25, 2**-5.00, 2**-4.75, 2**-4.5, 2**-4.25, 2**-4, 2**-3.75, 2**-3.5, 2**-3.25, 2**-3]
fine_gamma_range = [2**1, 2**1.25, 2**1.50, 2**1.75, 2**2, 2**2.25, 2**2.50, 2**2.75, 2**3.00, 2**3.25, 2**3.50, 2**3.75, 2**4, 2**4.25, 2**4.5, 2**4.75, 2**5]
# Setting the gammaCScores list to empty to record the new c, gamma, accuracy tuples
gammaCScores = []
for i in fine_c_range:
    for j in fine_gamma_range:
        gammaCfinder(i, j)
        
meanScores = []
maxgammaCScore = [-1,-1,-1]
for val in gammaCScores:
    meanScores.append(val[2])
    if(val[2] > maxgammaCScore[2]):
        maxgammaCScore = val
        
print("Maximum Accuracy (fine grid search): ", max(meanScores))
print("Maximum Accuracy (c, gamma, max. accuracy values) (fine grid search): ", maxgammaCScore)

# Converting the list of accuracies to sets to identify the regions with same accuracy
s = set(meanScores)

# There are 6 different (distinct) accuracies obtained
print("Distinct sets (fine grid search): ", s)
print("Length: ", len(s))

# Task 8
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter([val[0] for val in gammaCScores], [val[1] for val in gammaCScores], [val[2]*100 for val in gammaCScores], c=meanScores, cmap='Set1', marker='o', s=[5]*len(meanScores))
ax.set_xlabel("C")
ax.set_ylabel("gamma")
ax.set_zlabel("% Accuracy")
plt.show()