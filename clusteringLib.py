import random
import numpy as np

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
            for pointId in testSet:
                proximity=[]
                for centr in centroids:
                    proximity.append(np.linalg.norm(np.subtract(centr,pointsDic[pointId])))
                probab.append(min(proximity))
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
        for pointId in testSet:
            distances=[]
            for centr in centroids:
                distances.append(np.linalg.norm(np.subtract(centr,pointsDic[pointId])))
            clusterIds[distances.index(min(distances))].append(pointId)
        for i in range(k):
            if clusterIds[i]==[]:
                centroids[i]=pointsDic[random.sample(testSet,1)[0]]    #Readjust centroid of empty cluster
            else:
                centroids[i]=list(np.mean([pointsDic[id] for id in clusterIds[i]],axis=0))
        if oldClusterIds==clusterIds:
            break
    return clusterIds

def clusterMetrics(clusterIds,pointsDic):
    meanList=[]
    varList=[]
    for cluster in clusterIds:
        clusterPoints=[]
        for l in cluster:
            clusterPoints.append(pointsDic[l])
        meanList.append(np.mean(clusterPoints,axis=0))
        varList.append(np.var(clusterPoints,axis=0))
    return meanList,varList