import numpy as np
import matplotlib.pyplot as plt
import random
from groupTourLib import *
import sys
from joblib import Parallel, delayed

city = "Rome"
graph = readGraph(city)
users = readUsers(city)
pois = readPOIs(city,len(graph))
stayTime = readStayTimes(city,len(graph))

########## K-MEANS CLUSTERING ##########
########################################
def kmeans(k, testSet, pointsDic, init='random'):
    
    if init=='random':
        initSet = random.sample(testSet, k) #Forgy initialization
        centroids = [pointsDic[i] for i in initSet]
    elif init=='kmeans++':
        centroids=[]
        centroids.append(pointsDic[random.sample(testSet,1)[0]])
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
            centroids.append(pointsDic[testSet[ind]])
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
        with open('log.txt','a') as f:
            f.write(str(rep)+' '+str(k)+'\n')
        clusterIds = kmeans(k,list(pois.keys()),pois,init='kmeans++')
        userCat={}
        for userId in sampleUsers:
            userCat[userId]=[]
        for ind,cluster in enumerate(clusterIds): 
            clusterUsers=usersInRadius(cluster,pois,1.1,sampleUsers)
            for user in clusterUsers:
                userCat[user].append(ind)
        catUsers={}
        for ind in range(k):
            catUsers[ind]=[]
        for userId in userCat:
            if userCat[userId]==[]:
                catUsers[random.randint(0,k-1)].append(userId)
            else:
                catUsers[random.randint(0,len(userCat[userId])-1)].append(userId)
        clusterProfits=[]
        for ind,cluster in enumerate(clusterIds):
            testUsers=[]
            for l in catUsers[ind]:
                testUsers.append(sampleUsers[l])
            if testUsers==[]:
                clusterProfits.append(0)
            else:
                testPOIs={}
                for l in cluster:
                    testPOIs[l]=pois[l]
                clusterPath = bestRatioPlusPath(s, t, testUsers,graph,scoring,stayTime,B,testPOIs)
                clusterProfits.append(pathProfit(clusterPath, testUsers, scoring, testPOIs))
        kscores.append(sum(clusterProfits))
    return kscores

klist=[1,2,3,4,5]
m=100
B=420
totalReps=200
scoring='sum'
results=[]
numOfCores=int(sys.argv[1])

if numOfCores==1:
    for rep in range(totalReps):
        results.append(categoryRep(rep))
else:
    results = Parallel(n_jobs=numOfCores)(delayed(categoryRep)(rep) for rep in range(totalReps))

print(results)