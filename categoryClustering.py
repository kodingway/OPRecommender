import numpy as np
import matplotlib.pyplot as plt
import random
from groupTourLib import *
import sys
from joblib import Parallel, delayed
from clusteringLib import *

city = "Rome"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

def usersInRadius(testSet,pois,mult,users):
    clusterPOIs=[]
    for poiId in testSet:
        clusterPOIs.append(pois[poiId])
    mean = np.mean(clusterPOIs,axis=0)

    clusterRadius = 0
    for poiId in testSet:
        dist = np.linalg.norm(np.subtract(mean,pois[poiId]))
        if dist > clusterRadius:
            clusterRadius=dist
    
    clusterUsers=[]
    for userId in users:
        dist = np.linalg.norm(np.subtract(mean,users[userId]))
        if dist <= mult*clusterRadius:
            clusterUsers.append(userId)
    return clusterUsers

def categoryRep(rep):
    with open('log.txt','a') as f:
        f.write(str(rep)+'\n')
    
    while True:
        (s, t) = random.sample(range(1,len(graph)), 2)
        if pathCost([s,t],stayTime,graph) <= B:
            break
    
    sampleIds = random.sample(list(users.keys()), m)
    sampleUsers={}
    for userId in sampleIds:
        sampleUsers[userId]=users[userId]
    kscores=[]
    for k in klist:
        clusterIds = kmeans(k,list(pois.keys()),pois,init='kmeans++')
        meanList,_ = clusterMetrics(clusterIds,pois)
        catUsers={}
        for ind in range(k):
            catUsers[ind]=[]
        for userId in sampleUsers:
            dist=[]
            for ind,centr in enumerate(meanList):
                dist.append(np.linalg.norm(np.subtract(centr,sampleUsers[userId])))
            catUsers[dist.index(min(dist))].append(userId)
        clusterProfits=[]
        for ind,cluster in enumerate(clusterIds):
            testUsers=[]
            for l in catUsers[ind]:
                testUsers.append(sampleUsers[l])
            if testUsers==[]:
                clusterProfits.append(0)
            else:
                clusterPath = bestRatioPlusPath(s, t, testUsers,graph,scoring,stayTime,B,pois)
                clusterProfits.append(pathProfit(clusterPath, testUsers, scoring, pois))
        kscores.append(sum(clusterProfits))
    return kscores

klist=[1,2,5,10,20,50,100]
m=100
B=420
totalReps=500
scoring='sum'
results=[]
numOfCores=int(sys.argv[1])

if numOfCores==1:
    for rep in range(totalReps):
        results.append(categoryRep(rep))
else:
    results = Parallel(n_jobs=numOfCores)(delayed(categoryRep)(rep) for rep in range(totalReps))

with open('categoryClusteringRome100.dat','w') as f:
    f.write(str(results))