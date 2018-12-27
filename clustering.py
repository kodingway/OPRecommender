import random
import numpy as np
from groupTourLib import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed

city = "Rome"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

########## K-MEANS CLUSTERING ##########
########################################
def kmeans(k, testSet, users, init='random'):
    
    if init=='random':
        initSet = random.sample(testSet, k) #Forgy initialization
        centroids = [users[i] for i in initSet]
    elif init=='kmeans++':
        centroids=[]
        centroids.append(users[random.sample(testSet,1)[0]])
        while len(centroids)<k:
            probab=[]
            for userId in testSet:
                userProximity=[]
                for centr in centroids:
                    userProximity.append(np.linalg.norm(np.subtract(centr,users[userId])))
                probab.append(min(userProximity))
            probab=[x/sum(probab) for x in probab]
            probab=np.cumsum(probab)
            randNum = random.random()
            for i,p in enumerate(probab):
                if randNum<p:
                    ind=i
                    break
            centroids.append(users[testSet[ind]])
    clusterIds=[]

    for i in range(k):
        clusterIds.append([])
    while True:
        oldClusterIds = clusterIds.copy()
        clusterIds=[]
        for i in range(k):
            clusterIds.append([])
        for userId in testSet:
            distances=[]
            for centr in centroids:
                distances.append(np.linalg.norm(np.subtract(centr,users[userId])))
            clusterIds[distances.index(min(distances))].append(userId)
        for i in range(k):
            if clusterIds[i]==[]:
                centroids[i]=users[random.sample(testSet,1)[0]]    #Readjust centroid of empty cluster
            else:
                centroids[i]=list(np.mean([users[id] for id in clusterIds[i]],axis=0))
        if oldClusterIds==clusterIds:
            break
    return clusterIds

def clusterMetrics(clusterIds,users):
    meanList=[]
    varList=[]
    for cluster in clusterIds:
        clusterUsers=[]
        for l in cluster:
            clusterUsers.append(users[l])
        meanList.append(np.mean(clusterUsers,axis=0))
        varList.append(np.var(clusterUsers,axis=0))
    return meanList,varList

m=100 #Number of users
totalScores = []
B=420
scoring = 'sum'
totalReps=10
########## VISUALIZATION ##########
# pca = PCA(n_components=2).fit(list(pois.values()))
###################################

def clusteringRep(rep):
    with open('log.txt','a') as f:
        f.write(str(rep)+'\n')
    
    while True:
        (s, t) = random.sample(range(1,len(graph)), 2)
        if pathCost([s,t],stayTime,graph) <= B:
            break
    
    testSet = random.sample(range(1,len(users)+1), m)
    kscores=[]
    for k in [1,2,5,10,20]:
        # print(rep,k)
        clusterIds = kmeans(k,testSet,users,init='kmeans++')
        meanList, varList = clusterMetrics(clusterIds,users)
        # print(np.sum(varList,axis=1))
        ########## VISUALIZATION ##########
        # reduced_data = pca.transform(list(pois.values()))
        # plt.figure()
        # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
        # reduced_data = pca.transform([pois[s],pois[t]])
        # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='P')
        # for ind,cluster in enumerate(clusterIds):
        #     reduced_data = pca.transform([users[clusterId] for clusterId in cluster])
        #     plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10,marker='D')
        #     reduced_data = pca.transform([meanList[ind]])
        #     plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='x',c='k')
        #     plt.show()
        ###################################
        clusterProfits=[]
        # plt.figure()
        for ind,cluster in enumerate(clusterIds):
            # reduced_data = pca.transform(list(pois.values()))
            # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10)
            # reduced_data = pca.transform([users[clusterId] for clusterId in cluster])
            # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=10,marker='D')
            testUsers=[]
            for l in cluster:
                testUsers.append(users[l])
            testUsers = np.array([np.array(x) for x in testUsers])
            clusterPath = bestRatioPlusPath(s, t, testUsers,graph,scoring,stayTime,B,pois)
            # reduced_data = pca.transform([pois[poiId] for poiId in clusterPath])
            # plt.scatter(reduced_data[:,0],reduced_data[:,1],s=50,marker='P')
            # plt.show()
            print(pathProfit(clusterPath, testUsers, scoring, pois)/len(testUsers))
            clusterProfits.append(pathProfit(clusterPath, testUsers, scoring, pois))
            # plt.clf()
        kscores.append(sum(clusterProfits))
    return kscores

numOfCores=int(sys.argv[1])
if numOfCores==1:
    for rep in range(totalReps):
        totalScores.append(clusteringRep(rep))
else:
    totalScores = Parallel(n_jobs=numOfCores)(delayed(clusteringRep)(rep) for rep in range(totalReps))

with open('result.dat','w') as f:
    f.write(str(totalScores))