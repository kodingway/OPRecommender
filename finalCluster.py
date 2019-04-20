import random
import numpy as np
from groupTourLib import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from clusteringLib import *

city = "myFlorence"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

pca = PCA(n_components=2).fit(list(pois.values()))

k=3
bestDist=100
for rep in range(20):
    print(rep)
    clusterIds = kmeans(k,list(users.keys()),users,init='kmeans++')
    meanList, distList = clusterMetrics(clusterIds,users)
    if np.mean(distList)<bestDist:
        bestDist=np.mean(distList)
        bestClusterIds=clusterIds[:]

plt.figure()
reduced_data=pca.transform(list(pois.values()))
plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
for cluster in bestClusterIds:
    clusterUsers=[users[userId] for userId in cluster]
    reduced_data = pca.transform(clusterUsers)
    plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
plt.show()

with open('FlorenceClusters.txt','w') as f:
    for ind,cluster in enumerate(bestClusterIds):
        for userId in cluster:
            f.write(userId+' ')
        f.write('\n')