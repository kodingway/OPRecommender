import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from clusteringLib import *
from groupTourLib import *

from mpl_toolkits.mplot3d import Axes3D

cities = ['myRome','myFlorence','myPisa']
graph={}
users={}
pois={}
for city in cities:
    graph[city] = readGraph(city)
    users[city] = readUsers(city)
    pois[city] = readPOIs(city,len(graph))

commonUsers=[]
with open('Dataset/Rome+Florence.txt','r') as f:
    spamreader=csv.reader(f,delimiter=' ')
    for row in spamreader:
        commonUsers.append(row[0])

pca = PCA(n_components=2).fit(list(pois['myRome'].values())+list(pois['myFlorence'].values())+list(pois['myPisa'].values()))
clusterIds=kmeans(5,list(pois['myRome'].keys()),pois['myRome'],init='kmeans++')
plt.figure()
for cluster in clusterIds:
    clusterPOIs=[pois['myRome'][poiId] for poiId in cluster]
    reduced_data = pca.transform(clusterPOIs)
    plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
#     # plt.show()
plt.show()

plt.figure()
plt.title('Points of interest (Regularized)')
for city in cities:
    reduced_data = pca.transform(list(pois[city].values()))
    plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
    # reduced_data = pca.transform(list(users[city].values()))
    # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10,marker='x')
    # plt.show()
plt.legend(['Rome','Florence','Pisa'])
plt.show()
# plt.savefig('Cities-Reg.png')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# pca = PCA(n_components=3).fit(list(pois['myRome'].values())+list(pois['myFlorence'].values())+list(pois['myPisa'].values()))
# for city in cities[:2]:
#     reduced_data = pca.transform(list(pois[city].values()))
#     ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2],marker='o')
#     reduced_data = pca.transform([users[city][userId] for userId in commonUsers])
#     ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2],marker='D')
# # reduced_data = pca.transform(list(users['myFlorence'].values()))
# # ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2],marker='D')
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# pca = PCA(n_components=3).fit(list(pois['myRome'].values())+list(pois['myFlorence'].values())+list(pois['myPisa'].values()))
# clusterIds=kmeans(4,list(pois['myPisa'].keys()),pois['myPisa'],init='kmeans++')
# for cluster in clusterIds:
#     clusterPOIs=[pois['myPisa'][poiId] for poiId in cluster]
#     reduced_data = pca.transform(clusterPOIs)
#     ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])
#     # plt.show()
# plt.show()
