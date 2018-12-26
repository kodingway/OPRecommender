import random
import numpy as np
from groupTourLib import *
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
scoreFunc = satisfactionSum
totalReps=200

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
        print(rep,k)
        clusterIds = kmeans(k,testSet,users,init='kmeans++')
        meanList, varList = clusterMetrics(clusterIds,users)
        totalProfit=0
        for cluster in clusterIds:
            testUsers=[]
            for l in cluster:
                testUsers.append(users[l])
            testUsers = np.array([np.array(xi) for xi in testUsers])
            clusterPath = bestRatioPlusPath(s, t, testUsers,graph,scoreFunc,stayTime,B,pois)
            totalProfit += pathProfit(clusterPath, testUsers, scoreFunc, pois)
        kscores.append(totalProfit)
    return kscores

results  = Parallel(n_jobs=20)(delayed(clusteringRep)(rep) for rep in range(totalReps))
totalScores = [res[0] for res in results]

with open('result.dat','w') as f:
    f.write(str(totalScores))